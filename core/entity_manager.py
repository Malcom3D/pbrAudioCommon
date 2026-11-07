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
import copy
import threading
import numpy as np
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, List

from ..utils.config import Config
from ..lib.klepto_entity_store import KleptoEntityStore
from ..lib.debug_utils import debug_print, set_debug, set_debug_prefix

# Collections that should be fully hydrated at init-time, and always kept hot.
_EAGER_COLLECTIONS = frozenset({"sources", "objects", "outputs"})

# Singletons always kept in _singleton dict (not persisted together).
_EAGER_SINGLETONS = frozenset({
    "config", "frames", "sample_counter", "connected_buffer",
    "frequency_bands", "soxel_grid", "geometry_data",
    "material_properties", "medium_properties",
})

class EntityManager:
    _instance = None
    _initialized = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config: str, *, hot_cache_size: int = 512):
        if self._initialized:
            return

        self._sources: Dict[int, Any] = {}
        self._objects: Dict[int, Any] = {}
        self._outputs: Dict[int, Any] = {}
        self._output_datas: Dict[int, Any] = {}
        self._wave_propagators: Dict[int, Any] = {}
        self._layer_managers: Dict[int, Any] = {}
        self._trajectories: Dict[int, Any] = {}
        self._collisions: Dict[int, Any] = {}
        self._forces: Dict[int, Any] = {}
        self._modal_vertices: Dict[int, Any] = {}
        self._score_tracks: Dict[int, Any] = {}
        self._fracture_events: Dict[int, Any] = {}
        self._rigidbody_synth: Dict[int, Any] = {}
        self._resonance_synth: Dict[int, Any] = {}
        self._singleton: Dict[str, Any] = {}

        # Per-collection locks for thread-safe register/unregister
        self._locks: Dict[str, threading.RLock] = {
            coll: threading.RLock() for coll in (
                "sources", "objects", "outputs", "output_datas",
                "wave_propagators", "trajectories", "collisions",
                "forces", "modal_vertices", "score_tracks",
                "fracture_events", "rigidbody_synth", "resonance_synth",
            )
        }

        self._initialized = True

        self.sigleton_map = {
            "config": "Config",
            "frames": "FrameCounter",
            "sample_counter": "SampleCounter",
            "connected_buffer": "ConnectedConnectedBuffer",
            "frequency_bands": "FrequencyBands",
            "soxel_grid": "SoxelGrid",
            "geometry_data": "GeometryData",
            "material_properties": "MaterialProperties",
            "medium_properties": "MediumProperties",
        }
        self.entities_map = {
            "sources": ["ParticlesSource", "ObjectSource", "EnvironmetSource", "SphericalSource", "PlanarSource"],
            "objects": ["AcousticObject", "SurfaceVoxelObject"],
            "outputs": ["AmbisonicOutput", "OmnidirectionalOutput", "Figure8Output", "CardioidOutput", "HypercardioidOutput"],
            "wave_propagators": "WavePropagator",
            "output_datas": "OutputData",
            "trajectories": ["ParticlesTrajectoryData", "TrajectoryData", "tmpTrajectoryData"],
            "collisions": ["CollisionData", "ParticlesCollisionsPoints", "ParticlesCollisionsVoxels"],
            "forces": ["ForceData", "ForceDataSequence"],
            "modal_vertices": "ModalVertices",
            "score_tracks": "ScoreTrack",
            "fracture_events": "FractureEvent",
            "rigidbody_synth": "RigidBodySynth",
            "resonance_synth": "ResonanceSynth",
        }

        # Build config + store
        self._config = Config(config)
        self.register("config", self._config)

        set_debug(self._config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self._store = KleptoEntityStore(self._config)

        # LRU hot cache for collections *not* in _EAGER_COLLECTIONS.
        self._hot: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()
        self._hot_cache_size = hot_cache_size

        # ------------------------------------------------------------------
               # Hydrate eager collections + singletons from disk
        # ------------------------------------------------------------------
        self._hydrate_eager()

        # ------------------------------------------------------------------
        # Let ResumeData load the rest (it also calls register())
        # ------------------------------------------------------------------
        try:
            from ..lib.resume_data import ResumeData
            ResumeData(self).load_data(eager_only=False)
        except Exception as e:
            debug_print(f"[EntityManager] ResumeData skipped: {e}")

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------
    def _hydrate_eager(self):
        """Load eager collections/singletons fully into RAM."""
        for name in _EAGER_COLLECTIONS:
            if not self._store.has_entity(name):
                continue
            for idx in self._store.iter_keys(name):
                obj = self._store.get(name, idx)
                if obj is None:
                    continue
                with self._locks[name]:
                    getattr(self, f"_{name}")[idx] = obj
            debug_print(f"[EntityManager] hydrated {name} {self.count_entity(name)[name]} items)")

        for name in _EAGER_SINGLETONS:
            if name == "config":
                continue  # already registered
            obj = self._store.get(name, 0)
            if obj is not None:
                self._singleton[name] = obj

    # ------------------------------------------------------------------
    # Register / unregister
    # ------------------------------------------------------------------
    def register(self, entity: str, obj: Any) -> Optional[int]:
        # --- singleton ---
        if entity in self.sigleton_map:
            self._singleton[entity] = obj
            self._store.put_sync(entity, 0, obj)
            return None

        # --- collection ---
        coll = self._resolve_collection(entity)
        if coll is None:
            debug_print(f"[EntityManager] register: unknown entity '{entity}'")
            return None

        if not self._matches(coll, obj):
            debug_print(f"[EntityManager] register: type mismatch "
                        f"for '{entity}' ({type(obj).__name__})")
            return None

        lock = self._locks[coll]
        with lock:
            container: Dict[int, Any] = getattr(self, f"_{coll}")
            idx = (max(container.keys()) + 1) if container else 0

            container[idx] = obj
            self._hot_put((coll, idx), obj)

        # Persist outside of the lock (I/O is slow)
        self._store.put_sync(coll, idx, obj)
        return idx

    def unregister(self, entity: str, idx: int = None) -> None:
        if entity in self.sigleton_map:
            if entity in self._singleton:
                del self._singleton[entity]
                self._store.delete(entity, 0)
            return

        coll = self._resolve_collection(entity)
        if coll is None or idx is None:
            return

        with self._locks[coll]:
            container = getattr(self, f"_{coll}")
            container.pop(idx, None)
            self._hot.pop((coll, idx), None)

        self._store.delete(coll, idx)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, entity: str = None, idx: int = None) -> Any:
        if entity is None:
            return (
                self._singleton, self._sources, self._objects, self._outputs,
                self._wave_propagators, self._output_datas, self._trajectories,
                self._collisions, self._forces, self._modal_vertices,
                self._score_tracks, self._fracture_events,
                self._rigidbody_synth, self._resonance_synth,
            )

        # Singleton?
        if entity in self.sigleton_map:
            if entity not in self._singleton:
                cached = self._store.get(entity, 0)
                if cached is not None:
                    self._singleton[entity] = cached
            cached = self._singleton.get(entity)
            if entity in ("geometry_data", "material_properties", "medium_properties"):
                return copy.deepcopy(cached)
            return cached

        # Collection?
        coll = self._resolve_collection(entity)
        if coll is None:
            return None

        container = getattr(self, f"_{coll}")

        if idx is None:
            # hydrate-on-demand for lazy collections returning whole dict
            if coll not in _EAGER_COLLECTIONS:
                with self._locks[coll]:
                    for key in self._store.iter_keys(coll):
                        if key not in container:
                            obj = self._store.get(coll, key)
                            if obj is not None:
                                container[key] = obj
                                self._hot_put((coll, key), obj)
            return container

        # Single index
        if idx in container:
            return container[idx]

        # Not hot — try hot cache, then disk
        hit = self._hot_get((coll, idx))
        if hit is not None:
            with self._locks[coll]:
                container[idx] = hit
            return hit

        obj = self._store.get(coll, idx)
        if obj is not None:
            with self._locks[coll]:
                container[idx] = obj
            self._hot_put((coll, idx), obj)
        return obj

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def count_entity(self, entity: str = None) -> Dict[str, int]:
        if entity is None:
            out = {}
            for name in self.sigleton_map:
                out[name] = 1 if name in self._singleton else 0
            for name in self.entities_map:
                out[name] = self._store_count(name)
            return out

        if entity in self.sigleton_map:
            return {entity: 1 if entity in self._singleton else 0}
        if entity in self.entities_map:
            return {entity: self._store_count(entity)}
        return {}

    def _store_count(self, coll: str) -> int:
        # store is the truth; but if lazy we may have unsaved writes →
        # add len(container) when container is preferred
        return len(self._store.iter_keys(coll)) or len(getattr(self, f"_{coll}", {}))

    # ------------------------------------------------------------------
    # Flush / close
    # ------------------------------------------------------------------
    def dump(self, is_score_track_final: bool = False):
        """Flush all hot / in-memory collections to klepto."""
        for name in self.entities_map:
            container: Dict[int, Any] = getattr(self, f"_{name}", {})
            if not container:
                continue
            # only write keys not yet in store
            new_items = [(k, v) for k, v in container.items()
                         if not self._store.exists(name, k)]
            if new_items:
                self._store.put_sync_many(name, new_items)

        for name, obj in self._singleton.items():
            if not self._store.exists(name, 0):
                self._store.put_sync(name, 0, obj)

        self._store.flush()
        debug_print("[EntityManager] dump complete")

    def close(self):
        self.dump()
        self._store.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_collection(self, entity: str) -> Optional[str]:
        for key in self.entities_map:
            if entity in key:
                return key
        return None

    def _matches(self, coll: str, obj: Any) -> bool:
        tname = str(type(obj))
        for sub in self.entities_map[coll] if isinstance(self.entities_map[coll], list) \
                else [self.entities_map[coll]]:
            if sub in tname:
                return True
        return False

    # ---- LRU hot cache ------------------------------------------------
    def _hot_get(self, key: Tuple[str, int]) -> Any:
        if key in self._hot:
            self._hot.move_to_end(key)
            return self._hot[key]
        return None

    def _hot_put(self, key: Tuple[str, int], value: Any):
        self._hot[key] = value
        self._hot.move_to_end(key)
        while len(self._hot) > self._hot_cache_size:
            self._hot.popitem(last=False)
