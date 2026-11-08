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
_BLOSC2_BOOL_THRESHOLD_BYTES = 64 * 1024       # sparse bool matrices
_BLOSC2_FLOAT_THRESHOLD_BYTES = 256 * 1024     # 4x higher; floats don't
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
