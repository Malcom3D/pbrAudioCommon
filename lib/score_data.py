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

import os
import re
import gzip
import json
import blosc2
import tarfile
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

# 16 MiB threshold for switching from in-memory numpy to blosc2 NDArray storage.
_BLOSC2_THRESHOLD_BYTES = 16 * 1024 * 1024  # 16 MiB


def _to_blosc2(tag: str, arr: Union[np.ndarray, blosc2.NDArray, None]) -> Union[np.ndarray, blosc2.NDArray, None]:
    """
    Return `arr` as a persistent `blosc2.NDArray` if it contains more than
    `_BLOSC2_THRESHOLD_BYTES` of payload, otherwise return it unchanged.

    For reading back, we tag the NDArray in its metadata (`tag`) so the
    loader can decide whether to keep it as NDArray (large) or convert it
    to a numpy array (small), even when the threshold changes later.
    """
    if arr is None:
        return None
    if isinstance(arr, blosc2.NDArray):
        return arr
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.nbytes <= _BLOSC2_THRESHOLD_BYTES:
        return arr

    # Create a specific cparams/dparams for score payloads.
    cparams = blosc2.CParams(
        codec=blosc2.Codec.LZ4, typesize=1, clevel=1, nthreads=8,
    )
    dparams = blosc2.DParams(nthreads=16)
    nd = blosc2.asarray(arr, cparams=cparams, dparams=dparams)
    nd.vlmeta["score_tag"] = tag
    return nd

def _from_blosc2(arr: Union[np.ndarray, blosc2.NDArray, None]) -> Union[np.ndarray, blosc2.NDArray, None]:
    """
    Keep large payloads as `blosc2.NDArray`, convert small ones to numpy.
    """
    if arr is None:
        return None
    if not isinstance(arr, blosc2.NDArray):
        return arr
    if arr.nbytes <= _BLOSC2_THRESHOLD_BYTES:
        return arr[:]
    return arr

@dataclass
class ScoreEvent:
    """Represents the score for a single collision"""
    coll_obj: int
    start_sample: int
    stop_sample: int
    type: np.ndarray # shape(total_samples,1, dtype=np.int32)
    vertex_ids: Union[blosc2.ndarray.NDArray, np.ndarray]  # shape(total_samples,n_vertices)
    contact_area: Union[blosc2.ndarray.NDArray, np.ndarray] = None # shape(total_samples,1)
    force: Union[blosc2.ndarray.NDArray, np.ndarray] = None # shape(total_samples,1) Excitation force magnitude
    coupling_data: Union[blosc2.ndarray.NDArray, np.ndarray] = None # shape(total_samples,1) Array of [other_obj_idx, coupling_strength] pairs

    def __post_init__(self):
        # Enforce the ">16MB -> blosc2 NDArray" policy on every construction,
        # including those coming from `ScoreTrack.load()`.
        self.type = _to_blosc2("type", self.type) if not isinstance(self.type, blosc2.NDArray) else self.type
        self.vertex_ids = _to_blosc2("vertex_ids", self.vertex_ids)
        self.contact_area = _to_blosc2("contact_area", self.contact_area)
        self.force = _to_blosc2("force", self.force)
        self.coupling_data = _to_blosc2("coupling_data", self.coupling_data)

    def _get(self, name):
        v = getattr(self, name)
        if isinstance(v, blosc2.NDArray):
            return v[:]
        return v

    def __getstate__(self):
        state = self.__dict__.copy()
        # blosc2 NDArrays don't pickle reliably; ship raw numpy to the pickler.
        for k in ("type", "vertex_ids", "contact_area", "force", "coupling_data"):
            v = state.get(k)
            if isinstance(v, blosc2.NDArray):
                state[k] = v[:]  # materialize
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()  # re-apply the >16MB -> NDArray policy

    def get_event_at_sample(self, sample_idx: int):
        """Get all events at a specific sample index."""
        return self.type[sample_idx][0], np.unique(np.nonzero(self.vertex_ids[sample_idx])).tolist(), self.force[sample_idx], self.contact_area[sample_idx], self.coll_obj, self.coupling_data[sample_idx]

    def save(self, output_path: str, obj_idx: int, idx: int):

        score_event = {
            'coll_obj': self.coll_obj,
            'start_sample': self.start_sample,
            'stop_sample': self.stop_sample,
        }

        filenames = []

        json_file = f"{obj_idx}_{self.coll_obj}_{idx}.json"
        with open(f"{output_path}/{json_file}", 'w') as f:
            json.dump(score_event, f, indent=2)
        filenames.append(json_file)

        def _save_field(name: str, arr, suffix: str):
            if arr is None:
                return
            filename = f"{obj_idx}_{self.coll_obj_obj}_{idx}_{name}.{suffix}"
            if isinstance(arr, blosc2.NDArray):
                # Keep the existing container (cparams/dparams preserved).
                blosc2.save(arr, f"{output_path}/{filename}", mode="w")
            else:
                # Small numpy payload: still wrap it in a blosc2 container.
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
