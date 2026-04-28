"""Dask/Distributed optimization: inlined fast functions and task annotations."""

import _operator
import functools as fct
import re
from typing import List, Optional, Union

import dask.array as darr
import numpy as np
import zarr as zr
from dask.core import flatten
from dask.optimization import cull, fuse, inline, inline_functions
from dask.utils import ensure_dict
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import SchedulerState, cast


ANNOTATIONS = {
    "from-zarr-store": {"resources": {"MEM": 1}},
    "load_avi_ffmpeg": {"resources": {"MEM": 1}},
    "est_motion_chunk": {"resources": {"MEM": 1}},
    "transform_perframe": {"resources": {"MEM": 0.5}},
    "pnr_perseed": {"resources": {"MEM": 0.5}},
    "ks_perseed": {"resources": {"MEM": 0.5}},
    "smooth_corr": {"resources": {"MEM": 1}},
    "vectorize_noise_fft": {"resources": {"MEM": 1}},
    "vectorize_noise_welch": {"resources": {"MEM": 1}},
    "update_spatial_block": {"resources": {"MEM": 1}},
    "tensordot_restricted": {"resources": {"MEM": 1}},
    "update_temporal_block": {"resources": {"MEM": 1}},
    "merge_restricted": {"resources": {"MEM": 1}},
}
# Patterns -> resource annotations (:doc:`distributed:resources`).
# Task names matched by regex keys; used by :class:`TaskAnnotation`.


def _fast_functions_vindex() -> list:
    """Helpers for inlined fancy indexing — API differs across Dask releases."""
    c = darr.core
    if hasattr(c, "_vindex_slice_and_transpose"):
        return [
            c._vindex_merge,
            c._vindex_slice_and_transpose,
        ]
    out = []
    for name in ("_vindex_merge", "_vindex_slice", "_vindex_transpose"):
        fn = getattr(c, name, None)
        if fn is not None:
            out.append(fn)
    return out


# See :doc:`dask:optimize` — inlined “fast functions” during graph optimization.
FAST_FUNCTIONS = [
    darr.core.getter_inline,
    darr.core.getter,
    _operator.getitem,
    zr.core.Array,
    darr.chunk.astype,
    darr.core.concatenate_axes,
    *_fast_functions_vindex(),
]


class TaskAnnotation(SchedulerPlugin):
    """
    Custom `SchedulerPlugin` that implemented per-task level annotation. The
    annotations are applied according to the module constant
    :const:`ANNOTATIONS`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.annt_dict = ANNOTATIONS

    def update_graph(self, scheduler, client, tasks, **kwargs):
        parent = cast(SchedulerState, scheduler)
        for tk in tasks.keys():
            for pattern, annt in self.annt_dict.items():
                if re.search(pattern, tk):
                    ts = parent._tasks.get(tk)
                    res = annt.get("resources", None)
                    if res:
                        ts._resource_restrictions = res
                    pri = annt.get("priority", None)
                    if pri:
                        pri_org = list(ts._priority)
                        pri_org[0] = -pri
                        ts._priority = tuple(pri_org)


def custom_arr_optimize(
    dsk: dict,
    keys: list,
    fast_funcs: list = FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict: Optional[dict] = None,
    rewrite_dict: Optional[dict] = None,
    keep_patterns=[],
) -> dict:
    """
    Customized implementation of array optimization function.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_funcs : list, optional
        List of fast functions to be inlined. By default :const:`FAST_FUNCTIONS`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    rename_dict : dict, optional
        Dictionary mapping old task keys to new ones. Only used during fusing of
        tasks. By default `None`.
    rewrite_dict : dict, optional
        Dictionary mapping old task key substrings to new ones. Applied at the
        end of optimization to all task keys. By default `None`.
    keep_patterns : list, optional
        List of patterns of task keys that should be preserved during
        optimization. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.

    See Also
    -------
    :doc:`dask:optimize`
    `dask.array.optimization.optimize`
    """
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk


def rewrite_key(key: Union[str, tuple], rwdict: dict) -> str:
    """
    Rewrite a task key according to `rwdict`.

    Parameters
    ----------
    key : Union[str, tuple]
        Input task key.
    rwdict : dict
        Dictionary mapping old task key substring to new ones. All keys in this
        dictionary that exists in input `key` will be substituted.

    Returns
    -------
    key : str
        The new key.

    Raises
    ------
    ValueError
        if input `key` is neither `str` or `tuple`
    """
    typ = type(key)
    if typ is tuple:
        k = key[0]
    elif typ is str:
        k = key
    else:
        raise ValueError("key must be either str or tuple: {}".format(key))
    for pat, repl in rwdict.items():
        k = re.sub(pat, repl, k)
    if typ is tuple:
        ret_key = list(key)
        ret_key[0] = k
        return tuple(ret_key)
    else:
        return k


def custom_fused_keys_renamer(
    keys: list, max_fused_key_length=120, rename_dict: Optional[dict] = None
) -> str:
    """
    Custom implmentation to create new keys for `fuse` tasks.

    Uses custom `split_key` implementation.

    Parameters
    ----------
    keys : list
        List of task keys that should be fused together.
    max_fused_key_length : int, optional
        Used to limit the maximum string length for each renamed key. If `None`,
        there is no limit. By default `120`.
    rename_dict : dict, optional
        Dictionary used to rename keys during fuse. By default `None`.

    Returns
    -------
    fused_key : str
        The fused task key.

    See Also
    -------
    split_key
    dask.optimization.fuse
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)

    if max_fused_key_length:  # Take into account size of hash suffix
        max_fused_key_length -= 5

    def _enforce_max_key_limit(key_name):
        if max_fused_key_length and len(key_name) > max_fused_key_length:
            name_hash = f"{hash(key_name):x}"[:4]
            key_name = f"{key_name[:max_fused_key_length]}-{name_hash}"
        return key_name

    if typ is str:
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key)
        concatenated_name = "-".join(names)
        return _enforce_max_key_limit(concatenated_name)
    elif typ is tuple and len(first_key) > 0 and isinstance(first_key[0], str):
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key[0])
        concatenated_name = "-".join(names)
        return (_enforce_max_key_limit(concatenated_name),) + first_key[1:]


def split_key(key: Union[tuple, str], rename_dict: Optional[dict] = None) -> str:
    """
    Split, rename and filter task keys.

    This is custom implementation that only keeps keys found in :const:`ANNOTATIONS`.

    Parameters
    ----------
    key : Union[tuple, str]
        The input task key.
    rename_dict : dict, optional
        Dictionary used to rename keys. By default `None`.

    Returns
    -------
    new_key : str
        New key.
    """
    if type(key) is tuple:
        key = key[0]
    kls = key.split("-")
    if rename_dict:
        kls = list(map(lambda k: rename_dict.get(k, k), kls))
    kls_ft = list(filter(lambda k: k in ANNOTATIONS.keys(), kls))
    if kls_ft:
        return "-".join(kls_ft)
    else:
        return kls[0]


def check_key(key: Union[str, tuple], pat: str) -> bool:
    """
    Check whether `key` contains pattern.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat : str
        Pattern to check.

    Returns
    -------
    bool
        Whether `key` contains pattern.
    """
    try:
        return bool(re.search(pat, key))
    except TypeError:
        return bool(re.search(pat, key[0]))


def check_pat(key: Union[str, tuple], pat_ls: List[str]) -> bool:
    """
    Check whether `key` contains any pattern in a list.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat_ls : List[str]
        List of pattern to check.

    Returns
    -------
    bool
        Whether `key` contains any pattern in the list.
    """
    for pat in pat_ls:
        if check_key(key, pat):
            return True
    return False


def inline_pattern(dsk: dict, pat_ls: List[str], inline_constants: bool) -> dict:
    """
    Inline tasks whose keys match certain patterns.

    Parameters
    ----------
    dsk : dict
        Input dask graph.
    pat_ls : List[str]
        List of patterns to check.
    inline_constants : bool
        Whether to inline constants.

    Returns
    -------
    dsk : dict
        Dask graph with keys inlined.

    See Also
    -------
    dask.optimization.inline
    """
    keys = [k for k in dsk.keys() if check_pat(k, pat_ls)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants)
        for k in keys:
            del dsk[k]
        if inline_constants:
            dsk, dep = cull(dsk, set(list(flatten(keys))))
    return dsk


def custom_delay_optimize(
    dsk: dict, keys: list, fast_functions=[], inline_patterns=[], **kwargs
) -> dict:
    """
    Custom optimization functions for delayed tasks.

    By default only fusing of tasks will be carried out.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_functions : list, optional
        List of fast functions to be inlined. By default `[]`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.
    """
    dsk, _ = fuse(ensure_dict(dsk), rename_keys=custom_fused_keys_renamer)
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if fast_functions:
        dsk = inline_functions(
            dsk,
            [],
            fast_functions=fast_functions,
        )
    return dsk


def unique_keys(keys: list) -> np.ndarray:
    """
    Returns only unique keys in a list of task keys.

    Dask task keys regarding arrays are usually tuples representing chunked
    operations. This function ignore different chunks and only return unique keys.

    Parameters
    ----------
    keys : list
        List of dask keys.

    Returns
    -------
    unique : np.ndarray
        Unique keys.
    """
    new_keys = []
    for k in keys:
        if isinstance(k, tuple):
            new_keys.append("chunked-" + k[0])
        elif isinstance(k, str):
            new_keys.append(k)
    return np.unique(new_keys)


def get_keys_pat(pat: str, keys: list, return_all=False) -> Union[list, str]:
    """
    Filter a list of task keys by pattern.

    Parameters
    ----------
    pat : str
        Pattern to check.
    keys : list
        List of keys to be filtered.
    return_all : bool, optional
        Whether to return all keys matching `pat`. If `False` then only the
        first match will be returned. By default `False`.

    Returns
    -------
    keys : Union[list, str]
        If `return_all` is `True` then a list of keys will be returned.
        Otherwise only one key will be returned.
    """
    keys_filt = list(filter(lambda k: check_key(k, pat), list(keys)))
    if return_all:
        return keys_filt
    else:
        return keys_filt[0]


