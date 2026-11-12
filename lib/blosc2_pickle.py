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
Helpers to embed blosc2 NDArrays inside pickle streams safely.

We tag a bytes object with a 6-byte magic header so __setstate__ can
recognize it and rebuild the NDArray without needing numpy→blosc2 in the
pickler path. This is critical because:

  - blosc2.NDArray doesn't pickle reliably (holds mmap/C handles).
  - We *want* compression to survive serialization, so we cannot just
    call `arr[:]` — that defeats the whole point for sparse arrays.
  - We don't want the KleptoEntityStore to have to sniff object internals.

Usage:
    >>> payload = pack_ndarray(ndarray)
    >>> ...  # ships as bytes in pickle
    >>> arr = unpack_ndarray(payload)  # optional: fallback if tag missing
"""

import pickle
from typing import Any, Iterator

import numpy as np
import blosc2

# 6-byte magic: "B2NARR" (Blosc2 NDArray). Chosen to be unique enough
# that a pickled bytes payload of arbitrary content won't collide.
_NDARRAY_MAGIC = b"B2NARR"
_NDARRAY_HEADER_LEN = len(_NDARRAY_MAGIC) + 4  # magic + 4-byte payload len

# Blosc2 settings tuned for sparse bool / occasional float arrays.
_DEFAULT_CPARAMS = dict(codec=blosc2.Codec.LZ4, typesize=1, clevel=1, nthreads=8)
_DEFAULT_DPARAMS = dict(nthreads=16)

def _ndarray_to_blob(nd: blosc2.NDArray) -> bytes:
    """
    Serialize an in-memory blosc2.NDArray to a self-contained bytes blob.

    `blosc2.asarray()` (which we use below) keeps the compressed chunks in
    splitted form; `nd.schunk.to_cframe()` gives us a single contiguous
    buffer we can ship through pickle. On read, `blosc2.schunk_from_cframe`
    rebuilds the exact same container — compression preserved.
    """
    # nd.schunk is the low-level schunk. to_cframe() is the canonical
    # way to get a serializable buffer that keeps compression settings.
    return nd.schunk.to_cframe()

def _blob_to_ndarray(blob: bytes) -> blosc2.NDArray:
    """Inverse of _ndarray_to_blob. Returns an in-memory NDArray."""
    schunk = blosc2.schunk_from_cframe(blob, copy=False)
    # blosc2.NDArray wraps an existing schunk when passed positionally.
    return blosc2.NDArray(schunk)

def pack_ndarray(nd: blosc2.NDArray) -> bytes:
    """
    Return a tagged bytes blob that pickle can ship.

    Format: b"B2NARR" + <u32 le payload length> + <blosc2 cframe>
    The length lets a future reader verify framing, and (most importantly)
    makes the tag impossible to confuse with a plain pickled blob.
    """
    payload = _ndarray_to_blob(nd)
    return _NDARRAY_MAGIC + len(payload).to_bytes(4, "little") + payload

def is_packed_ndarray(obj: Any) -> bool:
    """True if `obj` is a blob produced by `pack_ndarray()`."""
    if not isinstance(obj, (bytes, bytearray, memoryview)):
        return False
    head = bytes(obj[:_NDARRAY_MAGIC.__len__()])
    return head == _NDARRAY_MAGIC

def unpack_ndarray(blob: bytes) -> blosc2.NDArray:
    """Inverse of `pack_ndarray()`. Raises ValueError on bad framing."""
    if not is_packed_ndarray(blob):
        raise ValueError("Not a B2NARR blob")
    hdr = len(_NDARRAY_MAGIC)
#    (length,) = int.from_bytes(blob[hdr : hdr + 4], "little"),
    length = int.from_bytes(blob[hdr : hdr + 4], "little")
    payload = bytes(blob[hdr + 4 : hdr + 4 + length])
    if len(payload) != length:
        raise ValueError(f"B2NARR payload length mismatch: got {len(payload)}, expected {length}")
    return _blob_to_ndarray(payload)


# ---------------------------------------------------------------------------
# Convenience: numpy fallback for small arrays, NDArray for large ones
# ---------------------------------------------------------------------------

# Anything smaller than this stays as a numpy array (compression overhead
# of blosc2's chunk scheme isn't worth it for tiny payloads).
_BLOSC2_THRESHOLD_BYTES = 64 * 1024  # 64 KiB

def to_maybe_ndarray(arr: Any, *, threshold_bytes: int = _BLOSC2_THRESHOLD_BYTES, pack_bools: bool = True, cparams: dict = None, dparams: dict = None) -> Any:
    """
    Return `arr` unchanged if small, or a blosc2.NDArray if it exceeds
    `threshold_bytes`.

    If `pack_bools` and the array is bool, it is packed with `np.packbits`
    first (8x reduction), then wrapped in blosc2 — **but only if the result
    still exceeds `threshold_bytes` after packing**. This is the big win
    for `vertex_ids`, which is a `(samples, vertices)` bool matrix.

    `pack_bools=True` is safe here because callers (ScoreEvent) know the
    semantic of each field.
    """
    if arr is None:
        return None
    if isinstance(arr, blosc2.NDArray):
        return arr
    if not isinstance(arr, np.ndarray):
        return arr

    # Try bool packing first; this can take a huge bool array below the
    # blosc2 threshold, so we *keep it as numpy* — pickle compression will
    # still kick in for free (via _pickle_bz2 streaming, see store).
    if pack_bools and arr.dtype == np.bool_:
        packed = np.packbits(arr, axis=-1)  # (samples, ceil(vertices/8))
        # Stash original width so consumers can unpack losslessly.
        # We use a dict-like "record" so it survives pickle as plain bytes.
        if packed.nbytes >= threshold_bytes:
            cparams = cparams or _DEFAULT_CPARAMS
            dparams = dparams or _DEFAULT_DPARAMS
            nd = blosc2.asarray(packed, cparams=cparams, dparams=dparams)
            # Tag so callers know how to decode.
            nd.vlmeta["encoding"] = "packbits_bool"
            nd.vlmeta["orig_shape"] = list(arr.shape)
            return nd
        # Still below blosc2 threshold after packbits → keep as numpy.
        # Wrap in a carrier so pickle round-trips the bool semantics.
        return _PackedBool(packed, list(arr.shape))

    # Non-bool: plain size-cutoff.
    if arr.nbytes >= threshold_bytes:
        cparams = cparams or _DEFAULT_CPARAMS
        dparams = dparams or _DEFAULT_DPARAMS
        return blosc2.asarray(arr, cparams=cparams, dparams=dparams)
    return arr

def from_maybe_ndarray(obj: Any) -> Any:
    """
    Inverse of `to_maybe_ndarray()`. Returns a numpy array in every case.

    - blosc2.NDArray → materialized numpy
    - _PackedBool → unpacked bool numpy
    - numpy / None / other → returned as-is-is
    """
    if isinstance(obj, blosc2.NDArray):
        enc = obj.vlmeta.get("encoding", None)
        if enc == "packbits_bool":
            orig_shape = tuple(obj.vlmeta["orig_shape"])
            packed = obj[:]
            return np.unpackbits(packed, axis=-1, count=orig_shape[-1]).astype(bool)
        return obj[:]
    if isinstance(obj, _PackedBool):
        return np.unpackbits(obj.data, axis=-1, count=obj.orig_shape[-1]).astype(bool)
    return obj

# ---------------------------------------------------------------------------
# _PackedBool: tiny pickle-friendly carrier for sub-threshold bool arrays
# ---------------------------------------------------------------------------

class _PackedBool:
    """
    Internal-only carrier: a bool array that is < 64 KiB after packbits.

    We can't make it a picklable subclass easily with bitarray semantics,
    and we don't want to lose the type info ("this was bool") through
    pure bytes. This wrapper holds:
      - `data`       : uint8 packed buffer
      - `orig_shape` : list[int]
    """
    __slots__ = ("data", "orig_shape")

    def __init__(self, data: np.ndarray, orig_shape: list):
        self.data = data
        self.orig_shape = orig_shape

    def __getstate__(self):
        return {"data": self.data, "orig_shape": self.orig_shape}

    def __setstate__(self, state):
        self.data = state["data"]
        self.orig_shape = state["orig_shape"]

    def __repr__(self):
        return f"_PackedBool(shape={tuple(self.orig_shape)}, dtype={self.data.dtype})"

# ---------------------------------------------------------------------------
# A Pickler subclass that ships blosc2.NDArray as tagged blobs
# ---------------------------------------------------------------------------

class Blosc2AwarePickler(pickle.Pickler):
    """
    Pickler that turns any blosc2.NDArray into a `B2NARR...` bytes blob.

    Why not use `reducer_override`? Because that only fires for the *outer*
    call of `__reduce_ex__`, and it wouldn't help for NDArrays buried inside
    a `__getstate__` dict. Instead we hook `persistent_id`? No — that
    requires a persistent store.

    The correct hook is `reducer_override`. It's called for every object
    that goes through the pickle reducer path, at every depth. That's
    exactly what we need: nested NDArrays inside ScoreEvent get turned into
    byte blobs transparently.

    Note: for this to fire, `ScoreEvent.__getstate__` must NOT eagerly
    convert blosc2 to numpy. If it does, we never get a chance to touch it.
    """

    def reducer_override(self, obj):
        # Hook every object that passes through.
        if isinstance(obj, blosc2.NDArray):
            return (
                unpack_ndarray,           # callable
                (pack_ndarray(obj),),     # args (single bytes)
                None,                     # state (unused)
                None,                     # listitems
                None,                     # dictitems
            )
        # Returning NotImplemented lets the default machinery handle it.
        return NotImplemented


class Blosc2AwareUnpickler(pickle.Unpickler):
    """
    We don't actually need to override `find_class` for pure B2NARR blobs
    (they're just bytes on the wire). But we *do* want a symmetric wrapper
    so that future stream-format tweaks have a single place to live.
    """

    # Intentionally empty: unpack_ndarray is still called from the pickler
    # stream as a global reference, which the pickle module resolves via
    # import. Keep the class for symmetry and future hooks.
    pass

# ---------------------------------------------------------------------------
# One-shot helpers used by KleptoEntityStore
# ---------------------------------------------------------------------------

def dumps(obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
    import io
    buf = io.BytesIO()
    Blosc2AwarePickler(buf, protocol=protocol).dump(obj)
    return buf.getvalue()

def loads(blob: bytes) -> Any:
    import io
    return Blosc2AwareUnpickler(io.BytesIO(blob)).load()
