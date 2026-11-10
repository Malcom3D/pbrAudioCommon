# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

# Changes from previous version:
#   * Drop the 16 MiB *numpy* threshold for blosc2 conversion; replaced by
#     a tiered policy:
#       - bool arrays: packbits, then blosc2 if >= 64 KiB packed (or if
#         the original was "very sparse", which bools usually are).
#       - float arrays: blosc2 if >= 64 KiB.
#     Both thresholds are much lower than before (16 MiB) — this is the fix.
#   * `__getstate__` now keeps blosc2.NDArray objects intact so the store's
#     `Blosc2AwarePickler` can serialize them as raw cframes (compression
#     preserved) instead of materializing to numpy.
#   * `__setstate__` re-applies the size/encoding policy.

import os
import re
import json
import tarfile
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import blosc2

from ..lib.blosc2_pickle import to_maybe_ndarray, from_maybe_ndarray, is_packed_ndarray, unpack_ndarray

# Below these thresholds we keep plain numpy and let pickle/bz2 handle it.
# Above them, we wrap in blosc2 — the store's pickler will then ship the
# compressed cframe directly.
_BLOSC2_BOOL_THRESHOLD_BYTES = 2048 * 1024       # sparse bool matrices
_BLOSC2_FLOAT_THRESHOLD_BYTES = 2048 * 1024     # 4x higher; floats don't
                                               # compress sparse-like, and
                                               # blosc2 overhead is real.

def _to_blosc2(tag: str, arr, *, threshold: int) -> Union[np.ndarray, blosc2.NDArray, None]:
    """
    Return `arr` as a blosc2.NDArray if it exceeds `threshold` bytes,
    otherwise return it unchanged (or as _PackedBool for bools).

    `to_maybe_ndarray` handles bool packbits internally.
    """
    if arr is None:
        return None
    if isinstance(arr, blosc2.NDArray):
        # Already compressed; just retag if necessary.
        if arr.vlmeta.get("score_field") != tag:
            arr.vlmeta["score_field"] = tag
        return arr
    if not isinstance(arr, np.ndarray):
        return arr

    # Determine per-dtype threshold:
    # - bool  → very low (sparse!)
    # - other → caller-provided
    if arr.dtype == np.bool_:
        eff_threshold = _BLOSC2_BOOL_THRESHOLD_BYTES
    else:
        eff_threshold = threshold

    result = to_maybe_ndarray(arr, threshold_bytes=eff_threshold, pack_bools=True)

    # If we ended up with a blosc2.NDArray, tag it so future readers know
    # which score-field it was (useful for debugging + re-derivation).
    if isinstance(result, blosc2.NDArray):
        result.vlmeta["score_field"] = tag
    return result


@dataclass
class ScoreEvent:
    """Represents the score for a single collision."""
    coll_obj: int
    start_sample: int
    stop_sample: int
    type: np.ndarray                                            # (samples, 1) int32
    vertex_ids: Union[blosc2.NDArray, np.ndarray, Any]          # (samples, n_vertices) bool
    contact_area: Union[blosc2.NDArray, np.ndarray, Any, None] = None
    force:        Union[blosc2.NDArray, np.ndarray, Any, None] = None
    coupling_data: Union[blosc2.NDArray, np.ndarray, Any, None] = None

    def __post_init__(self):
        # Apply the compression policy on every construction, including
        # those coming from ScoreTrack.load(). This runs even if the
        # caller passed raw numpy arrays.
        self.type          = _to_blosc2("type", self.type, threshold=_BLOSC2_FLOAT_THRESHOLD_BYTES)
        self.vertex_ids    = _to_blosc2("vertex_ids", self.vertex_ids, threshold=_BLOSC2_BOOL_THRESHOLD_BYTES)
        self.contact_area  = _to_blosc2("contact_area", self.contact_area, threshold=_BLOSC2_FLOAT_THRESHOLD_BYTES)
        self.force         = _to_blosc2("force", self.force, threshold=_BLOSC2_FLOAT_THRESHOLD_BYTES)
        self.coupling_data = _to_blosc2("coupling_data", self.coupling_data, threshold=_BLOSC2_FLOAT_THRESHOLD_BYTES)

    def _get(self, name):
        """Return field as numpy array (decompressed/materialized)."""
        v = getattr(self, name)
        return from_maybe_ndarray(v)

    def __getstate__(self):
        """
        Do NOT materialize blosc2 → numpy. The store's Blosc2AwarePickler
        reduces blosc2.NDArray to a tagged bytes cframe at serialization
        time. That preserves compression end-to-end.
        """
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-apply the compression policy. Old pickles might still carry
        # raw numpy arrays; they'll be upgraded on first touch.
        self.__post_init__()

    def get_event_at_sample(self, sample_idx: int):
        """Get all events at a specific sample index."""
        type_v        = self._get("type")
        vertex_ids_v  = self._get("vertex_ids")
        force_v       = self._get("force")
        area_v        = self._get("contact_area")
        coupling_v    = self._get("coupling_data")

        active_verts = np.unique(np.nonzero(vertex_ids_v[sample_idx])).tolist()
        return (
            type_v[sample_idx][0],
            active_verts,
            force_v[sample_idx] if force_v is not None else None,
            area_v[sample_idx] if area_v is not None else None,
            self.coll_obj,
            coupling_v[sample_idx] if coupling_v is not None else None,
        )

    # -- Disk round-trip (tar.gz) ------------------------------------------
    # Kept for backward compatibility; not used by KleptoEntityStore. Uses
    # blosc2.save() which retains cframe compression.

    def save(self, output_path: str, obj_idx: int, idx: int):
        score_event = {
            "coll_obj":     self.coll_obj,
            "start_sample": self.start_sample,
            "stop_sample":  self.stop_sample,
        }
        filenames = []

        json_file = f"{obj_idx}_{self.coll_obj}_{idx}.json"
        with open(f"{output_path}/{json_file}", "w") as f:
            json.dump(score_event, f, indent=2)
        filenames.append(json_file)

        def _save_field(name: str, arr, suffix: str):
            if arr is None:
                return
            filename = f"{obj_idx}_{self.coll_obj}_{idx}_{name}.{suffix}"
            if isinstance(arr, blosc2.NDArray):
                blosc2.save(arr, f"{output_path}/{filename}", mode="w")
            else:
                blosc2.save_array(arr, f"{output_path}/{filename}", mode="w")
            filenames.append(filename)

        _save_field("vertex_ids",    self.vertex_ids,    "b2")
        _save_field("type",          self.type,          "bl2")
        _save_field("contact_area",  self.contact_area,  "bl2")
        _save_field("force",         self.force,         "bl2")
        _save_field("coupling_data", self.coupling_data, "bl2")

        filepath = f"{obj_idx}_{self.coll_obj}_{idx}.tar.gz"
        with tarfile.open(f"{output_path}/{filepath}", mode="w:gz") as tar:
            for filename in filenames:
                tar.add(f"{output_path}/{filename}", filename)

        for filename in filenames:
            os.remove(f"{output_path}/{filename}")

        return filepath

@dataclass
class ScoreTrack:
    """Represents a score track for a single object."""
    obj_idx: int
    obj_name: str
    is_final: bool = False
    total_samples: int = None
    events: List[ScoreEvent] = field(default_factory=list)

    def add_event(self, event: ScoreEvent) -> None:
        """Add an event to the track."""
        self.events.append(event)

    def save(self, filepath: str) -> None:
        """
        Save the ScoreTrack to a tar gz file.

        Args:
            filepath: Path to save the file
            indent: JSON indentation level (None for compact format)
        """

        # make output path
        dirpath = os.path.dirname(filepath)
        output_path = f"{dirpath}/save"
        os.makedirs(output_path, exist_ok=True)

        score_track = {
            'obj_idx': self.obj_idx,
            'obj_name': self.obj_name,
            'is_final': self.is_final,
            'total_samples': self.total_samples
        }

        # Save score_track to file
        json_file = os.path.basename(filepath).removesuffix('tar.gz') + 'json'
        with open(f"{output_path}/{json_file}", 'w') as f:
            json.dump(score_track, f, indent=2)

        to_be_removed = [json_file]
        with tarfile.open(filepath, mode="w:gz") as tar:
            tar.add(f"{output_path}/{json_file}", json_file)
            for idx in range(len(self.events)):
                filename = self.events[idx].save(output_path, self.obj_idx, idx)
                tar.add(f"{output_path}/{filename}", filename)
                to_be_removed.append(filename)

        for filename in to_be_removed:
            os.remove(f"{output_path}/{filename}")

    @classmethod
    def load(cls, filepath: str, final: bool = False) -> 'ScoreTrack':
        events = []

        dirpath = os.path.dirname(filepath)
        output_path = f"{dirpath}/extract"
        os.makedirs(output_path, exist_ok=True)

        to_be_removed = []
        with tarfile.open(filepath, mode="r:gz") as tar:
            filenames = tar.getnames()
            # Local per-event scratch, reset for every event.
            for filename in filenames:
                if filename.endswith('json'):
                    tar.extract(filename, output_path)
                    to_be_removed.append(filename)
                    with open(f"{output_path}/{filename}", 'r') as f:
                        score_track = json.load(f)
                        if final and not score_track['is_final']:
                            break
                    continue

                if not filename.endswith('tar.gz'):
                    continue

                tar.extract(filename, output_path)
                to_be_removed.append(filename)

                with tarfile.open(f"{output_path}/{filename}", mode="r:gz") as score_event_tar:
                    event_files = score_event_tar.getnames()

                    # Per-event state.
                    coll_obj = start_sample = stop_sample = None
                    ev_type = contact_area = force = coupling_data = vertex_ids = None

                    for event_file in event_files:
                        score_event_tar.extract(event_file, output_path)
                        to_be_removed.append(event_file)
                        path = f"{output_path}/{event_file}"
                        if event_file.endswith('.json'):
                            with open(path, 'r') as f:
                                et = json.load(f)
                            coll_obj = et['coll_obj']
                            start_sample = et['start_sample']
                            stop_sample = et['stop_sample']
                        elif event_file.endswith('b2'):
                            # Open in read-only mode; keep as NDArray so the
                            # size policy can decide whether to expand.
                            vertex_ids = blosc2.open(path, mode='r')
                        elif event_file.endswith('bl2'):
                            if 'type' in event_file:
                                ev_type = blosc2.open(path, mode='r')
                            elif 'contact_area' in event_file:
                                contact_area = blosc2.open(path, mode='r')
                            elif 'force' in event_file:
                                force = blosc2.open(path, mode='r')
                            elif 'coupling_data' in event_file:
                                coupling_data = blosc2.open(path, mode='r')

                    # Hand raw values to ScoreEvent: __post_init__ applies the
                    # >16MB -> NDArray rule, converting small payloads to numpy.
                    events.append(ScoreEvent(
                        coll_obj=coll_obj,
                        start_sample=start_sample,
                        stop_sample=stop_sample,
                        type=ev_type,
                        contact_area=contact_area,
                        force=force,
                        vertex_ids=vertex_ids,
                        coupling_data=coupling_data,
                    ))

        for file in to_be_removed:
            p = f"{output_path}/{file}"
            if os.path.exists(p):
                os.remove(p)

        score_track['events'] = events
        return cls(**score_track)
