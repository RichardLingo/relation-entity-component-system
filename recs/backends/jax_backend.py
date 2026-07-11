"""JAX 后端实现。

需要安装 recs[jax]（即 jax>=0.4, jaxlib>=0.4）。
支持 JIT 编译与自动微分。

**重要限制**：JAX 数组是不可变的，而 RECS 的 EntityPool/Relation 大量使用原地修改操作。
因此 JAX 后端**仅支持只读查询操作**，不支持 add/remove/enable 等修改操作。
如需完整功能，请使用 NumPy/PyTorch/MLX 后端。

**macOS M 芯片支持**：
- JAX 通过 `jax-metal` 插件支持 Apple Silicon GPU (MPS)
- 需要安装: `pip install jax-metal`
- 默认使用 CPU，安装 jax-metal 后可自动使用 GPU
"""

import numpy as np

from .base import BackendBase


class JAXBackend(BackendBase):
    """JAX 后端：JIT 编译 + 自动微分。"""

    def __init__(self):
        import jax
        import jax.numpy as jnp
        
        # 启用 64 位模式以避免 int64 截断警告
        jax.config.update("jax_enable_x64", True)
        
        self._jnp = jnp

    # ---- dtype 映射 ----

    def _map_dtype(self, dtype):
        """将 NumPy dtype 映射为 JAX dtype。"""
        jnp = self._jnp
        if dtype is None:
            return jnp.float32
        # JAX 使用 NumPy dtype，直接转换
        return jnp.dtype(dtype)

    # ---- 元信息 ----

    @property
    def name(self) -> str:
        return 'jax'

    @property
    def device(self) -> str:
        import jax
        devices = jax.devices()
        if any(d.platform == 'gpu' for d in devices):
            return 'cuda'
        return 'cpu'

    # ---- 数组创建 ----

    def empty(self, shape, dtype):
        jnp = self._jnp
        if isinstance(shape, int):
            shape = (shape,)
        # JAX 没有 empty，用 zeros 代替
        return jnp.zeros(shape, dtype=self._map_dtype(dtype))

    def zeros(self, shape, dtype):
        jnp = self._jnp
        if isinstance(shape, int):
            shape = (shape,)
        return jnp.zeros(shape, dtype=self._map_dtype(dtype))

    def ones(self, shape, dtype):
        jnp = self._jnp
        if isinstance(shape, int):
            shape = (shape,)
        return jnp.ones(shape, dtype=self._map_dtype(dtype))

    def full(self, shape, fill_value, dtype):
        jnp = self._jnp
        if isinstance(shape, int):
            shape = (shape,)
        return jnp.full(shape, fill_value, dtype=self._map_dtype(dtype))

    def arange(self, start, stop=None, step=1, dtype=None):
        jnp = self._jnp
        if stop is None:
            stop = start
            start = 0
        return jnp.arange(start, stop, step, dtype=self._map_dtype(dtype))

    def array(self, data, dtype=None):
        jnp = self._jnp
        if isinstance(data, np.ndarray):
            arr = jnp.array(data)
        else:
            arr = jnp.array(data)
        if dtype is not None:
            arr = arr.astype(self._map_dtype(dtype))
        return arr

    # ---- 常用 dtype ----

    @property
    def int64(self):
        return self._jnp.int64

    @property
    def float32(self):
        return self._jnp.float32

    @property
    def float64(self):
        return self._jnp.float64

    @property
    def bool(self):
        return self._jnp.bool_

    # ---- 数组操作 ----

    def where(self, condition, x, y):
        return self._jnp.where(condition, x, y)

    def concatenate(self, arrays, axis=0):
        return self._jnp.concatenate(list(arrays), axis=axis)

    def copy(self, arr):
        # JAX 数组是不可变的，copy 是恒等操作
        return arr

    def asarray(self, obj, dtype=None):
        jnp = self._jnp
        if isinstance(obj, np.ndarray):
            arr = jnp.array(obj)
        elif hasattr(obj, '__jax_array__'):
            arr = obj.__jax_array__()
        else:
            arr = jnp.array(obj)
        if dtype is not None:
            arr = arr.astype(self._map_dtype(dtype))
        return arr

    def ascontiguousarray(self, arr):
        # JAX 数组默认连续
        return arr

    # ---- 索引与查询 ----

    def nonzero(self, arr):
        jnp = self._jnp
        result = jnp.nonzero(arr, size=None)
        if isinstance(result, tuple):
            return result[0]
        return result

    def boolean_mask(self, arr, mask):
        return arr[mask]

    def isin(self, elements, test_elements):
        jnp = self._jnp
        if not isinstance(test_elements, jnp.ndarray):
            test_elements = jnp.array(test_elements)
        if not isinstance(elements, jnp.ndarray):
            elements = jnp.array(elements)
        # 手动实现 isin
        flat = test_elements.flatten()
        result = elements.flatten()[:, None] == flat[None, :]
        return result.any(axis=-1).reshape(elements.shape)

    # ---- 聚合 ----

    def bincount(self, indices, weights=None, minlength=None):
        jnp = self._jnp
        if not isinstance(indices, jnp.ndarray):
            indices = jnp.array(indices)
        indices = indices.astype(jnp.int64)
        ml = minlength if minlength is not None else 0
        if weights is not None:
            if not isinstance(weights, jnp.ndarray):
                weights = jnp.array(weights)
            result = jnp.bincount(indices, weights=weights, length=ml)
        else:
            result = jnp.bincount(indices, length=ml)
        return result

    def add_at(self, output, indices, values):
        # JAX 数组是不可变的，add_at 返回新数组
        # 但为了接口一致，这里抛出异常提示使用 add_at_return
        raise NotImplementedError(
            "JAX 数组是不可变的，add_at 无法原地修改。"
            "请使用 result = backend.add_at_return(output, indices, values) 代替。"
        )

    def add_at_return(self, output, indices, values):
        """JAX 版本的 add_at，返回新数组而非原地修改。"""
        return output.at[indices].add(values)

    def sum(self, arr, axis=None):
        jnp = self._jnp
        if axis is None:
            return jnp.sum(arr)
        return jnp.sum(arr, axis=axis)

    def all(self, arr, axis=None):
        jnp = self._jnp
        if axis is None:
            return jnp.all(arr)
        return jnp.all(arr, axis=axis)

    # ---- 排序与唯一 ----

    def argsort(self, arr, kind='stable'):
        jnp = self._jnp
        return jnp.argsort(arr)

    def unique(self, arr):
        jnp = self._jnp
        return jnp.unique(arr)

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
        return np.asarray(arr)

    def from_numpy(self, arr):
        jnp = self._jnp
        return jnp.array(np.asarray(arr))

    # ---- 设备管理 ----

    def to_device(self, arr, device):
        import jax
        if device == 'cpu':
            return jax.device_put(arr, jax.devices('cpu')[0])
        elif device in ('cuda', 'gpu'):
            gpus = jax.devices('gpu')
            if gpus:
                return jax.device_put(arr, gpus[0])
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
        # JAX 不支持字符串数组
        return False

    def dtype_of(self, arr):
        if hasattr(arr, 'dtype'):
            return arr.dtype
        return np.asarray(arr).dtype
