"""NumPy 后端实现。

这是 RECS 的默认后端，始终可用，零额外依赖。
所有方法直接委托给 NumPy。
"""

import numpy as np

from .base import BackendBase


class NumpyBackend(BackendBase):
    """NumPy 后端：RECS 的默认计算后端。"""

    # ---- 元信息 ----

    @property
    def name(self) -> str:
        return 'numpy'

    @property
    def device(self) -> str:
        return 'cpu'

    # ---- 数组创建 ----

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype):
        return np.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype):
        return np.full(shape, fill_value, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        return np.arange(start, stop, step, dtype=dtype)

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    # ---- 常用 dtype ----

    @property
    def int64(self):
        return np.int64

    @property
    def float32(self):
        return np.float32

    @property
    def float64(self):
        return np.float64

    @property
    def bool(self):
        return np.bool_

    # ---- 数组操作 ----

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def copy(self, arr):
        return arr.copy()

    def asarray(self, obj, dtype=None):
        return np.asarray(obj, dtype=dtype)

    def ascontiguousarray(self, arr):
        return np.ascontiguousarray(arr)

    # ---- 索引与查询 ----

    def nonzero(self, arr):
        result = np.nonzero(arr)
        # 对于一维数组，返回第一个维度的索引
        if isinstance(result, tuple) and len(result) > 0:
            return result[0]
        return result

    def boolean_mask(self, arr, mask):
        return arr[mask]

    def isin(self, elements, test_elements):
        return np.isin(elements, test_elements)

    # ---- 聚合 ----

    def bincount(self, indices, weights=None, minlength=None):
        return np.bincount(indices, weights=weights, minlength=minlength or 0)

    def add_at(self, output, indices, values):
        np.add.at(output, indices, values)

    def sum(self, arr, axis=None):
        return np.sum(arr, axis=axis)

    def all(self, arr, axis=None):
        return np.all(arr, axis=axis)

    # ---- 排序与唯一 ----

    def argsort(self, arr, kind='stable'):
        return np.argsort(arr, kind=kind)

    def unique(self, arr):
        return np.unique(arr)

    # ---- 位操作 ----

    def packbits(self, arr, bitorder='little'):
        return np.packbits(np.asarray(arr, dtype=np.bool_), bitorder=bitorder)

    def unpackbits(self, arr, count, bitorder='little'):
        out = np.unpackbits(np.asarray(arr, dtype=np.uint8), bitorder=bitorder)
        return out[:count].astype(np.bool_, copy=False)

    # ---- 类型转换 ----

    def to_numpy(self, arr):
        return np.asarray(arr)

    def from_numpy(self, arr):
        return np.asarray(arr)

    # ---- 设备管理 ----

    def to_device(self, arr, device):
        # NumPy 只支持 CPU，忽略设备参数
        return arr

    # ---- dtype 工具 ----

    def is_floating(self, dtype):
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, ValueError):
            return False

    def is_integer(self, dtype):
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, ValueError):
            return False

    def is_bool(self, dtype):
        try:
            return np.issubdtype(dtype, np.bool_)
        except (TypeError, ValueError):
            return False

    def is_string(self, dtype):
        if isinstance(dtype, str) and dtype.startswith('U'):
            return True
        if isinstance(dtype, np.dtype) and dtype.kind == 'U':
            return True
        if dtype is str or dtype is object:
            return True
        return False

    def dtype_of(self, arr):
        return np.asarray(arr).dtype
