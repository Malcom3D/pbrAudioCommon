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
import gc
import pickle
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import numba as nb
import klepto
from dask import delayed, compute

## Synchronous scheduler only — no threads, no processes.
#dask_config.set(scheduler="synchronous")

# Configure Dask to use more threads
from dask import config as dask_config
#dask_config.set(scheduler='processes', num_workers=1024)
dask_config.set({'num_workers': 1024, 'optimization.fuse.active': True, 'optimization.fuse.max_depth': 10,})

from ..lib.debug_utils import debug_print, set_debug, set_debug_prefix


# ---------------------------------------------------------------------------
# Numba SIMD kernels — the only place heavy numeric work happens.
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def _pack_float32_arrays(arrays: List[np.ndarray], out: np.ndarray) -> int:
    """
    Copy a list of 1D float32 arrays into a single contiguous buffer.
    Returns the total number of elements written.
    SIMD-vectorised by LLVM; nogil so it never touches the interpreter.
    """
    offset = 0
    for i in range(len(arrays)):
        a = arrays[i]
        n = a.shape[0]
        for j in range(n):
            out[offset + j] = a[j]
        offset += n
    return offset


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def _unpack_float32_arrays(buf: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    """
    Inverse of _pack_float32_arrays. Returns a list of 1D views copied out.
    """
    n = lengths.shape[0]
    out = []
    offset = 0
    for i in range(n):
        L = lengths[i]
        a = np.empty(L, dtype=np.float32)
        for j in range(L):
            a[j] = buf[offset + j]
        out.append(a)
        offset += L
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def _pack_float64_arrays(arrays: List[np.ndarray], out: np.ndarray) -> int:
    offset = 0
    for i in range(len(arrays)):
        a = arrays[i]
        n = a.shape[0]
        for j in range(n):
            out[offset + j] = a[j]
        offset += n
    return offset


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def _hash_bytes(buf: np.ndarray) -> np.uint64:
    """FNV-1a 64-bit over a uint8 buffer. Used for cheap integrity checks."""
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    for i in range(buf.shape[0]):
        h ^= np.uint64(buf[i])
        h *= prime
    return h


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KleptoEntityStoreConfig:
    """Tuning knobs for KleptoEntityStore."""
    # Root directory for the persistent store.
    root: str = "./pbrAudioCache/entity_store"
    # klepto archive type: 'file' (one pickle per key) or 'dir' (dir per key).
    # 'file' is faster for many small objects; 'dir' for a few very large ones.
    archive_kind: str = "file"
    # If True, klepto keeps an in-memory mirror of everything (fast reads,
    # high memory). If False, every read hits disk (slow reads, low memory).
    cached: bool = False
    # Compress the pickle payloads? Trades CPU for disk.
    compress: bool = False
    # Use numba packing for float32/float64 array collections.
    use_numba_packing: bool = True
    # Minimum total bytes of a float array collection before we bother with
    # the numba path (below this, plain pickle is faster).
    numba_min_bytes: int = 1 << 16  # 64 KiB
    # fsync per write? Slower but crash-safe.
    fsync: bool = False
    # Debug
    debug: bool = False


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

class KleptoEntityStore:
    """
    Persistent, klepto-backed store for EntityManager dumps.

    Layout on disk:
        <root>/
            meta.json                       # schema + index
            <entity>/                       # e.g. 'trajectories', 'forces'
                <idx>.pkl                   # one file per entity (klepto file_archive)
            <entity>/_packed/<idx>.npz      # numba-packed numeric payloads
            <entity>/_packed/<idx>.meta.pkl # shape/length metadata for the packed payload

    The store is *append-friendly*: `put()` overwrites a single key's file,
    leaving every other key untouched. This is the main reason we prefer
    klepto.file_archive over a single big pickle.

    Threading/multiprocessing are never used. Heavy numeric work is done by
    numba `nogil` kernels; task structure is expressed with `dask.delayed`
    under the synchronous scheduler.
    """

    # Entity names recognised by EntityManager.dump() / ResumeData.load_data().
    DEFAULT_ENTITIES: Tuple[str, ...] = (
        "trajectories",
        "collisions",
        "forces",
        "modal_vertices",
        "score_tracks",
        "fracture_events",
        "rigidbody_synth",
        "resonance_synth",
    )

    def __init__(self, config: KleptoEntityStoreConfig):
        self.cfg = config
        self.root = os.path.abspath(config.root)
        os.makedirs(self.root, exist_ok=True)

        set_debug(config.debug)
        set_debug_prefix(self.__class__.__name__)

        # One klepto archive per entity. Lazily created on first access.
        self._archives: Dict[str, klepto.archives.file_archive] = {}
        self._archive_lock: Dict[str, bool] = {}

        # Meta index: entity -> {idx: {'format': 'pickle'|'packed', 'size': int}}
        self._meta: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._meta_path = os.path.join(self.root, "meta.json")
        self._load_meta()

    # ------------------------------------------------------------------ meta

    def _load_meta(self) -> None:
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "rb") as f:
                    raw = pickle.load(f)
                # JSON keys are strings; normalise ints.
                self._meta = {
                    ent: {int(k): v for k, v in per.items()}
                    for ent, per in raw.items()
                }
            except Exception as e:
                debug_print(f"meta load failed ({e}); starting empty")
                self._meta = {}
        else:
            self._meta = {}

    def _save_meta(self) -> None:
        # Atomic-ish write: write to tmp then rename.
        tmp = self._meta_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.d.dump(self._meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._meta_path)

    # --------------------------------------------------------------- archives

    def _archive_path(self, entity: str) -> str:
        return os.path.join(self.root, entity)

    def _get_archive(self, entity: str) -> klepto.archives.file_archive:
        if entity in self._archives:
            return self._archives[entity]

        path = self._archive_path(entity)
        os.makedirs(path, exist_ok=True)

        # klepto.file_archive writes one pickle per key. `cached=False` means
        # no in-memory mirror: every `__getitem__` reads from disk. This is
        # what we want for a large scene with many small objects.
        archive = klepto.archives.file_archive(
            path,
            dict=dict,
            serialized=True,   # pickle values on disk
            cached=self.cfg.cached,
        )
        self._archives[entity] = archive
        return archive

    # ------------------------------------------------------------- packing

    @staticmethod
    def _is_float_array_collection(obj: Any) -> bool:
        """True if obj is a list/tuple of 1D float32/float64 numpy arrays."""
        if not isinstance(obj, (list, tuple)) or len(obj) == 0:
            return False
        for a in obj:
            if not isinstance(a, np.ndarray):
                return False
            if a.ndim != 1:
                return False
            if a.dtype not in (np.float32, np.float64):
                return False
        return True

    def _pack_float_arrays(self, arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Flatten a list of 1D float arrays into one contiguous buffer.
        Returns (buffer, lengths, dtype_str).
        """
        dtype = arrays[0].dtype
        lengths = np.fromiter((a.shape[0] for a in arrays), dtype=np.int64, count=len(arrays))
        total = int(lengths.sum())
        buf = np.empty(total, dtype=dtype)

        if self.cfg.use_numba_packing and buf.nbytes >= self.cfg.numba_min_bytes:
            if dtype == np.float32:
                _pack_float32_arrays(arrays, buf)
            else:
                _pack_float64_arrays(arrays, buf)
        else:
            # Plain numpy path — still SIMD, still fast for small payloads.
            offset = 0
            for a in in arrays:
                n = a.shape[0]
                buf[offset:offset + n] = a
                offset += n

        return buf, lengths, str(dtype)

    @staticmethod
    def _unpack_float_arrays(buf: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
        if buf.dtype == np.float32:
            return _unpack_float32_arrays(buf, lengths)
        # float64 fallback (rare)
        out = []
        offset = 0
        for L in lengths:
            out.append(buf[offset:offset + L].copy())
            offset += L
        return out

    # ------------------------------------------------------------- put / get

    @delayed
    def _delayed_write_pickle(self, entity: str, idx: int, obj: Any) -> Tuple[str, int, int]:
        """Dask-delayed task: pickle one object to disk via klepto."""
        archive = self._get_archive(entity)
        archive[str(idx)] = obj
        # klepto.file_archive writes on assignment when cached=False.
        # If cached=True, we must explicitly dump.
        if self.cfg.cached:
            archive.dump()
        # Record size on disk.
        fpath = os.path.join(self._archive_path(entity), f"{idx}.pkl")
        size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        return entity, idx, size

    @delayed
    def _delayed_write_packed(self, entity: str, idx: int, arrays: List[np.ndarray]) -> Tuple[str, int, int]:
        """Dask-delayed task: numba-pack float arrays and write .npz + meta."""
        packed_dir = os.path.join(self._archive_path(entity), "_packed")
        os.makedirs(packed_dir, exist_ok=True)

        buf, lengths, dtype_str = self._pack_float_arrays(arrays)

        # Integrity hash (numba, nogil).
        h = int(_hash_bytes(buf.view(np.uint8)))

        npz_path = os.path.join(packed_dir, f"{idx}.npz")
        meta_path = os.path.join(packed_dir, f"{idx}.meta.pkl")

        np.savez(npz_path, buf=buf, lengths=lengths)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "dtype": dtype_str,
                    "n_arrays": len(arrays),
                    "total_elems": int(buf.shape[0]),
                    "hash": h,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        size = os.path.getsize(npz_path) + os.path.getsize(meta_path)
        return entity, idx, size

    def put(self, entity: str, idx: int, obj: Any) -> None:
        """
        Persist a single (entity, idx) -> obj.

        If obj is a list/tuple of 1D float arrays and is large enough, we
        take the numba-packed fast path. Otherwise we fall back to klepto's
        pickle path.
        """
        use_packed = (
            self.cfg.use_numba_packing
            and self._is_float_array_collection(obj)
        )

        if use_packed:
            # Estimate total bytes to decide if packing is worth it.
            total_bytes = sum(a.nbytes for a in obj)
            use_packed = total_bytes >= self.cfg.numba_min_bytes

        if use_packed:
            task = self._delayed_write_packed(entity, idx, list(obj))
            fmt = "packed"
        else:
            task = self._delayed_write_pickle(entity, idx, obj)
            fmt = "pickle"

        # Synchronous scheduler: this runs in the calling thread, no
        # threads/processes spawned.
        (ent, i, size), = compute(task)

        self._meta.setdefault(ent, {})[i] = {"format": fmt, "size": int(size)}
        self._save_meta()

    def put_many(self, entity: str, items: Iterable[Tuple[int, Any]]) -> None:
        """
        Persist many objects of the same entity. Builds one dask graph and
        computes it synchronously. Good for bulk dumps.
        """
        tasks = []
        formats: List[str] = []
        idxs: List[int] = []

        for idx, obj in items:
            use_packed = (
                self.cfg.use_numba_packing
                and self._is_float_array_collection(obj)
                and sum(a.nbytes for a in obj) >= self.cfg.numba_min_bytes
            )
            if use_packed:
                tasks.append(self._delayed_write_packed(entity, idx, list(obj)))
                formats.append("packed")
            else:
                tasks.append(self._delayed_write_pickle(entity, idx, obj))
                formats.append("pickle")
            idxs.append(idx)

        if not tasks:
            return

        results = compute(*tasks)
        for (ent, i, size), fmt in zip(results, formats):
            self._meta.setdefault(ent, {})[i] = {"format": fmt, "size": int(size)}
        self._save_meta()

    def get(self, entity: str, idx: int) -> Optional[Any]:
        """Load a single object. Returns None if not present."""
        per = self._meta.get(entity, {})
        info = per.get(idx)
        if info is None:
            # Fall back to klepto in case meta is stale.
            archive = self._get_archive(entity)
            if str(idx) in archive:
                return archive[str(idx)]
            return None

        if info["format"] == "packed":
            packed_dir = os.path.join(self._archive_path(entity), "_packed")
            npz_path = os.path.join(packed_dir, f"{idx}.npz")
            meta_path = os.path.join(packed_dir, f"{idx}.meta.pkl")
            if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
                return None
            with np.load(npz_path) as data:
                buf = data["buf"]
                lengths = data["lengths"]
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            # Optional integrity check.
            h = int(_hash_bytes(buf.view(np.uint8)))
            if h != meta["hash"]:
                debug_print(f"hash mismatch for {entity}/{idx} (continuing anyway)")
            # Ensure dtype matches what we wrote.
            if str(buf.dtype) != meta["dtype"]:
                buf = buf.astype(meta["dtype"])
            return self._unpack_float_arrays(buf, lengths)

        # pickle path
        archive = self._get_archive(entity)
        if str(idx) not in archive:
            return None
        return archive[str(idx)]

    def get_many(self, entity: str, idxs: Iterable[int]) -> Dict[int, Any]:
        """
        Load many objects. Uses dask.delayed to express the reads as a graph,
        executed by the synchronous scheduler.
        """
        idxs = list(idxs)
        tasks = [delayed(self.get)(entity, i) for i in idxs]
        results = compute(*tasks)
        return dict(zip(idxs, results))

    def delete(self, entity: str, idx: int) -> bool:
        per = self._meta.get(entity, {})
        if idx not in per:
            return False
        info = per.pop(idx)
        if info info["format"] == "packed":
            packed_dir = os.path.join(self._archive_path(entity), "_packed")
            for suffix in (".npz", ".meta.pkl"):
                p = os.path.join(packed_dir, f"{idx}{suffix}")
                if os.path.exists(p):
                    os.remove(p)
        else:
            p = os.path.join(self._archive_path(entity), f"{idx}.pkl")
            if os.path.exists(p):
                os.remove(p)
        self._save_meta()
        return True

    def keys(self, entity: str) -> List[int]:
        return sorted(self._meta.get(entity, {}).keys())

    def clear(self, entity: Optional[str] = None) -> None:
        if entity is None:
            for ent in list(self._meta.keys()):
                self.clear(ent)
            return
        path = self._archive_path(entity)
        if os.path.exists(path):
            shutil.rmtree(path)
        self._meta.pop(entity, None)
        self._archives.pop(entity, None)
        self._save_meta()

    # --------------------------------------------------- EntityManager bridge

    def dump_entity_manager(self, entity_manager_manager: Any, entities: Optional[Tuple[str, ...]] = None) -> None:
        """
        Persist every registered entity from an EntityManager.

        Uses the EntityManager's own getters so we never touch private state.
        """
        entities = entities or self.DEFAULT_ENTITIES

        for entity in entities:
            try:
                collection = entity_manager.get(entity)
            except Exception as e:
                debug_print(f"skip {entity}: {e}")
                continue

            if collection is None:
                continue

            # Singletons (e.g. 'config', 'frames') are dicts of name->obj.
            if not isinstance(collection, dict):
                # Wrap singletons as a single-key dict.
                self.put(entity, 0, collection)
                continue

            # Bulk write: build one dask graph per entity.
            items = [(int(k), v) for k, v in collection.items()]
            self.put_many(entity, items)

    def load_into_entity_manager(self, entity_manager: Any, entities: Optional[Tuple[str, ...]] = None) -> None:
        """
        Load persisted entities back into an EntityManager.

        This mirrors ResumeData.load_data() but reads from our store instead
        of the ad-hoc directories.
        """
        entities = entities or self.DEFAULT_ENTITIES

        for entity in entities:
            idxs = self.keys(entity)
            if not idxs:
                continue

            loaded = self.get_many(entity, idxs)
            for idx, obj in loaded.items():
                if obj is None:
                    continue
                try:
                    entity_manager.register(entity, obj)
                except Exception as e:
                    debug_print(f"register {entity}/{idx} failed: {e}")

    # --------------------------------------------------------------- lifecycle

    def flush(self) -> None:
        """Force any cached archives to disk. No-op when cached=False."""
        for archive in self._archives.values():
            try:
                archive.dump()
            except Exception as e:
                debug_print(f"flush failed: {e}")

    def close(self) -> None:
        self.flush()
        self._archives.clear()
        gc.collect()

    def __enter__(self) -> "KleptoEntityStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

