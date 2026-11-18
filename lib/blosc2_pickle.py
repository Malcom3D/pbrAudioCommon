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

"""
Picklable wrappers around blosc2 NDArray for use inside KleptoEntityStore.

The problem
-----------
blosc2.NDArray holds a reference to a live cframe, and pickling the object
directly is unreliable (the cframe isn't captured cleanly, and on unpickle the
file-format invariants may not hold).

The old workaround was `arr[:]` — full materialization — which defeats the
purpose of using blosc2 for large, mostly-empty boolean masks like
`ScoreEvent.vertex_ids`.

The solution
------------
Wrapper stores the *container format* in memory (`blosc2.NDArray.to_cframe`) and
rebuilds the NDArray via `blosc2.ndarray_from_cframe` on unpickle. nbytes, shape,
dtype, chunks, blocks and compression parameters all round-trip.
"""

from __future__ import annotations

import blosc2
import numpy as np
from typing import Any, Optional, Union


class PicklableBlosc2NDArray:
    """
    A minimal pickle-able proxy for `blosc2.NDArray`.

    Round-trip preserves:
        - shape, dtype
        - chunks, blocks
        - cparams / dparams
        - vlmeta (including user-defined tags like 'score_tag')

    The wrapper itself is a tiny Python object; the compressed payload lives in
    a single `bytes` attribute, so klepto effectively pickles one blob.
    """

    __slots__ = ("_buffer", "shape", "dtype", "cparams", "dparams", "vlmeta")

    def __init__(self, buffer: bytes, shape: tuple, dtype: np.dtype, cparams: Optional[dict] = None, dparams: Optional[dict] = None, vlmeta: Optional[dict] = None):
        self._buffer = buffer
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.cparams = dict(cparams) if cparams else None
        self.dparams = dict(dparams) if dparams else None
        self.vlmeta = dict(vlmeta) if vlmeta else {}

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_ndarray(cls, arr: blosc2.NDArray) -> "PicklableBlosc2NDArray":
        """
        Serialize a live `blosc2.NDArray` into a picklable wrapper.

        `blosc2.NDArray.to_cframe` returns the raw in-memory container bytes; on the
        other side we rebuild with `blosc2.ndarray_from_cframe`.
        """
        buf = blosc2.NDArray.to_cframe(arr)
        # cparams/dparams come back as dicts in recent blosc2 versions; keep
        # them defensive in case they're returned as CParams/DParams objects.
        try:
            cparams = dict(arr.cparams)
        except Exception:
            cparams = None
        try:
            dparams = dict(arr.dparams)
        except Exception:
            dparams = None

        try:
            vlmeta = dict(arr.vlmeta)
        except Exception:
            vlmeta = {}

        return cls(
            buffer=buf,
            shape=arr.shape,
            dtype=arr.dtype,
            cparams=cparams,
            dparams=dparams,
            vlmeta=vlmeta,
        )

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    def to_ndarray(self) -> blosc2.NDArray:
        """Rebuild the original `blosc2.NDArray` from the stored buffer."""
        # `blosc2.ndarray_from_cframe` accepts the same container produced by
        # `blosc2.NDArray.to_cframe`, so shape/dtype/cparams are inferred. Do NOT
        # pass shape/dtype here — they'd force densification.
        arr = blosc2.ndarray_from_cframe(self._buffer)
        # Re-attach vlmeta if any (options live in the serde header, but be
        # explicit in case the version of blosc2 drops them).
        for k, v in (self.vlmeta or {}).items():
            try:
                arr.vlmeta[k] = v
            except Exception:
                pass
        return arr

    # ------------------------------------------------------------------
    # Pickle protocol
    # ------------------------------------------------------------------
    def __getstate__(self):
        return {
            "buffer": self._buffer,
            "shape": self.shape,
            "dtype": self.dtype.str,
            "cparams": self.cparams,
            "dparams": self.dparams,
            "vlmeta": self.vlmeta,
        }

    def __setstate__(self, state):
        self._buffer = state["buffer"]
        self.shape = tuple(state["shape"])
        self.dtype = np.dtype(state["dtype"])
        self.cparams = state.get("cparams")
        self.dparams = state.get("dparams")
        self.vlmeta = state.get("vlmeta") or {}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __repr__(self) -> str:
        return (
            f"PicklableBlosc2NDArray(shape={self.shape}, "
            f"dtype={self.dtype}, nbytes={self.nbytes})"
        )


def wrap_for_pickle(value: Any) -> Any:
    """
    Recursively replace every `blosc2.NDArray` inside `value` with a
    `PicklableBlosc2NDArray`. Handles dataclasses, lists, tuples, dicts and
    bare NDArrays.
    """
    if isinstance(value, blosc2.NDArray):
        return PicklableBlosc2NDArray.from_ndarray(value)
    if isinstance(value, list):
        return [wrap_for_pickle(v) for v in value]
    if isinstance(value, tuple):
        return tuple(wrap_for_pickle(v) for v in value)
    if isinstance(value, dict):
        return {k: wrap_for_pickle(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            setattr(value, name, wrap_for_pickle(getattr(value, name)))
        return value
    return value


def unwrap_from_pickle(value: Any) -> Any:
    """Inverse of `wrap_for_pickle`."""
    if isinstance(value, PicklableBlosc2NDArray):
        return value.to_ndarray()
    if isinstance(value, list):
        return [unwrap_from_pickle(v) for v in value]
    if isinstance(value, tuple):
        return tuple(unwrap_from_pickle(v) for v in value)
    if isinstance(value, dict):
        return {k: unwrap_from_pickle(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            setattr(value, name, unwrap_from_pickle(getattr(value, name)))
        return value
    return value
