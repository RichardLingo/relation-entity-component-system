"""MLX 后端实现。

需要安装 recs[mlx]（即 mlx>=0.10）。
专为 Apple Silicon 统一内存架构设计。
"""

import numpy as np

from .base import BackendBase


class MLXBackend(BackendBase):
    """MLX 后端：Apple Silicon 统一内存，零拷贝。"""

    def __init__(self):
        import mlx.core as mx
        self._mx = mx

    # ---- dtype 映射 ----

    def _map_dtype(self, dtype):
        """将 NumPy dtype 映射为 MLX dtype。"""
        mx = self._mx
        if dtype is None:
            return mx.float32
        # 已经是 mlx dtype
        if hasattr(mx, 'Dtype') and isinstance(dtype, mx.Dtype):
            return dtype
        # NumPy dtype 或字符串
        dtype_str = str(np.dtype(dtype)) if not isinstance(dtype, str) else dtype
        mapping = {
            'float32': mx.float32,
            'float64': mx.float32,  # MLX 不支持 float64，降级
            'int64': mx.int32,  # MLX GPU scatter 不支持 int64，降级为 int32
            'int32': mx.int32,
            'int16': mx.int16,
            'int8': mx.int8,
            'bool': mx.bool_,
            'uint8': mx.uint8,
        }
        return mapping.get(dtype_str, mx.float32)

    # ---- 元信息 ----

    @property
    def name(self) -> str:
        return 'mlx'

    @property
    def device(self) -> str:
        return 'mps'  # Apple Silicon 统一内存

    # ---- 数组创建 ----

    def empty(self, shape, dtype):
        mx = self._mx
        if isinstance(shape, int):
            shape = (shape,)
        # MLX 没有 empty，用 zeros 代替
        return mx.zeros(shape, dtype=self._map_dtype(dtype))

    def zeros(self, shape, dtype):
        mx = self._mx
        if isinstance(shape, int):
            shape = (shape,)
        return mx.zeros(shape, dtype=self._map_dtype(dtype))

    def ones(self, shape, dtype):
        mx = self._mx
        if isinstance(shape, int):
            shape = (shape,)
        return mx.ones(shape, dtype=self._map_dtype(dtype))

    def full(self, shape, fill_value, dtype):
        mx = self._mx
        if isinstance(shape, int):
            shape = (shape,)
        return mx.full(shape, fill_value, dtype=self._map_dtype(dtype))

    def arange(self, start, stop=None, step=1, dtype=None):
        mx = self._mx
        if stop is None:
            stop = start
            start = 0
        return mx.arange(start, stop, step, dtype=self._map_dtype(dtype))

    def array(self, data, dtype=None):
        mx = self._mx
        if isinstance(data, np.ndarray):
            arr = mx.array(data)
        else:
            arr = mx.array(data)
        if dtype is not None:
            arr = arr.astype(self._map_dtype(dtype))
        return arr

    # ---- 常用 dtype ----

    @property
    def int64(self):
        return self._mx.int64

    @property
    def float32(self):
        return self._mx.float32

    @property
    def float64(self):
        return self._mx.float32  # MLX 不支持 float64

    @property
    def bool(self):
        return self._mx.bool_

    # ---- 数组操作 ----

    def where(self, condition, x, y):
        return self._mx.where(condition, x, y)

    def concatenate(self, arrays, axis=0):
        return self._mx.concatenate(list(arrays), axis=axis)

    def copy(self, arr):
        # MLX 数组是不可变的，copy 是恒等操作
        return arr

    def asarray(self, obj, dtype=None):
        mx = self._mx
        if isinstance(obj, mx.array.__class__):
            arr = obj
        elif isinstance(obj, np.ndarray):
            arr = mx.array(obj)
        else:
            arr = mx.array(obj)
        if dtype is not None:
            arr = arr.astype(self._map_dtype(dtype))
        return arr

    def ascontiguousarray(self, arr):
        # MLX 数组默认连续
        return arr

    # ---- 索引与查询 ----

    def nonzero(self, arr):
        mx = self._mx
        # MLX 没有可直接使用的 nonzero，用 NumPy 回退
        np_arr = np.asarray(arr)
        np_result = np.nonzero(np_arr)
        return mx.array(np_result[0])

    def boolean_mask(self, arr, mask):
        mx = self._mx
        # MLX 不支持布尔索引，统一使用整数索引
        idx = self.nonzero(mask)
        return arr[idx]

    def isin(self, elements, test_elements):
        mx = self._mx
        if not isinstance(test_elements, mx.array.__class__):
            test_elements = mx.array(test_elements)
        if not isinstance(elements, mx.array.__class__):
            elements = mx.array(elements)
        # 手动实现 isin
        flat = test_elements.flatten()
        result = elements.flatten()[:, None] == flat[None, :]
        return mx.any(result, axis=-1).reshape(elements.shape)

    # ---- 聚合 ----

    def bincount(self, indices, weights=None, minlength=None):
        mx = self._mx
        # MLX 没有 bincount，用 add_at 实现
        if not isinstance(indices, mx.array.__class__):
            indices = mx.array(indices)
        indices = indices.astype(mx.int64)
        ml = minlength if minlength is not None else 0
        if indices.size > 0:
            max_idx = int(mx.max(indices).item()) + 1
            ml = max(ml, max_idx)
        result = mx.zeros(ml, dtype=mx.float32 if weights is not None else mx.int64)
        if weights is not None:
            if not isinstance(weights, mx.array.__class__):
                weights = mx.array(weights)
            result = result.at[indices].add(weights)
        else:
            ones = mx.ones_like(indices, dtype=mx.int64)
            result = result.at[indices].add(ones)
        return result

    def add_at(self, output, indices, values):
        mx = self._mx
        # MLX 数组是不可变的，需要返回新数组
        # 但为了接口一致，这里修改 output 的引用
        # 注意：MLX 的 at[].add() 返回新数组，不是原地操作
        new_output = output.at[indices].add(values)
        # 无法原地修改，调用方需要处理
        # 这里抛出异常提示
        raise NotImplementedError(
            "MLX 数组是不可变的，add_at 无法原地修改。"
            "请使用 result = backend.add_at_return(output, indices, values) 代替。"
        )

    def add_at_return(self, output, indices, values):
        """MLX 版本的 add_at，返回新数组而非原地修改。"""
        return output.at[indices].add(values)

    def sum(self, arr, axis=None):
        mx = self._mx
        if axis is None:
            return mx.sum(arr)
        return mx.sum(arr, axis=axis)

    def all(self, arr, axis=None):
        mx = self._mx
        if axis is None:
            return mx.all(arr)
        return mx.all(arr, axis=axis)

    # ---- 排序与唯一 ----

    def argsort(self, arr, kind='stable'):
        mx = self._mx
        return mx.argsort(arr)

    def unique(self, arr):
        mx = self._mx
        # MLX 没有 unique，用排序 + 去重实现
        sorted_arr = mx.sort(arr)
        if sorted_arr.size == 0:
            return sorted_arr
        # 找到相邻不同的位置（转换为 numpy 处理）
        np_arr = np.array(sorted_arr)
        if np_arr.size == 0:
            return sorted_arr
        diff = np_arr[1:] != np_arr[:-1]
        mask = np.concatenate([np.array([True]), diff])
        # MLX 不支持布尔索引，用 take
        indices = np.nonzero(mask)[0]
        if len(indices) == 0:
            return mx.array([])
        return mx.take(sorted_arr, mx.array(indices))

    # ---- 位操作（委托给 NumPy） ----

    def packbits(self, arr, bitorder='little'):
        np_arr = self.to_numpy(arr).astype(np.bool_)
        result = np.packbits(np_arr, bitorder=bitorder)
        return self.from_numpy(result)

    def unpackbits(self, arr, count, bitorder='little'):
        np_arr = self.to_numpy(arr).astype(np.uint8)
        result = np.unpackbits(np_arr, bitorder=bitorder)[:count].astype(np.bool_)
        return self.from_numpy(result)

    # ---- 类型转换 ----

    def to_numpy(self, arr):
        if isinstance(arr, np.ndarray):
            return arr
        return np.array(arr)

    def from_numpy(self, arr):
        mx = self._mx
        return mx.array(np.asarray(arr))

    # ---- 设备管理 ----

    def to_device(self, arr, device):
        # MLX 统一内存，忽略设备参数
        return arr

    # ---- dtype 工具 ----

    def is_floating(self, dtype):
        mx = self._mx
        if hasattr(mx, 'Dtype') and isinstance(dtype, mx.Dtype):
            return dtype in (mx.float16, mx.float32)
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, ValueError):
            return False

    def is_integer(self, dtype):
        mx = self._mx
        if hasattr(mx, 'Dtype') and isinstance(dtype, mx.Dtype):
            return dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8)
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, ValueError):
            return False

    def is_bool(self, dtype):
        mx = self._mx
        if hasattr(mx, 'Dtype') and isinstance(dtype, mx.Dtype):
            return dtype == mx.bool_
        try:
            return np.issubdtype(dtype, np.bool_)
        except (TypeError, ValueError):
            return False

    def is_string(self, dtype):
        # MLX 不支持字符串张量
        return False

    def dtype_of(self, arr):
        mx = self._mx
        if isinstance(arr, mx.array.__class__):
            return arr.dtype
        return np.asarray(arr).dtype
