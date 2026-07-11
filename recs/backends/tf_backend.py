"""TensorFlow 后端实现。

需要安装 recs[tf]（即 tensorflow>=2.12）。
支持 Eager 模式与 Graph 模式。
"""

import numpy as np

from .base import BackendBase


class TFBackend(BackendBase):
    """TensorFlow 后端：支持 Eager / Graph 模式。"""

    def __init__(self):
        import tensorflow as tf
        self._tf = tf

    # ---- dtype 映射 ----

    def _map_dtype(self, dtype):
        """将 NumPy dtype 映射为 TensorFlow dtype。"""
        tf = self._tf
        if dtype is None:
            return tf.float32
        # 已经是 tf dtype
        if isinstance(dtype, tf.DType):
            return dtype
        # NumPy dtype 或字符串
        dtype_str = str(np.dtype(dtype)) if not isinstance(dtype, str) else dtype
        mapping = {
            'float32': tf.float32,
            'float64': tf.float64,
            'int64': tf.int64,
            'int32': tf.int32,
            'int16': tf.int16,
            'int8': tf.int8,
            'bool': tf.bool,
            'uint8': tf.uint8,
        }
        return mapping.get(dtype_str, tf.float32)

    # ---- 元信息 ----

    @property
    def name(self) -> str:
        return 'tensorflow'

    @property
    def device(self) -> str:
        tf = self._tf
        gpus = tf.config.list_physical_devices('GPU')
        return 'gpu' if gpus else 'cpu'

    # ---- 数组创建 ----

    def empty(self, shape, dtype):
        tf = self._tf
        if isinstance(shape, int):
            shape = (shape,)
        # TF 没有 empty，用 zeros 代替
        return tf.zeros(shape, dtype=self._map_dtype(dtype))

    def zeros(self, shape, dtype):
        tf = self._tf
        if isinstance(shape, int):
            shape = (shape,)
        return tf.zeros(shape, dtype=self._map_dtype(dtype))

    def ones(self, shape, dtype):
        tf = self._tf
        if isinstance(shape, int):
            shape = (shape,)
        return tf.ones(shape, dtype=self._map_dtype(dtype))

    def full(self, shape, fill_value, dtype):
        tf = self._tf
        if isinstance(shape, int):
            shape = (shape,)
        return tf.fill(shape, tf.constant(fill_value, dtype=self._map_dtype(dtype)))

    def arange(self, start, stop=None, step=1, dtype=None):
        tf = self._tf
        if stop is None:
            stop = start
            start = 0
        return tf.range(start, stop, step, dtype=self._map_dtype(dtype))

    def array(self, data, dtype=None):
        tf = self._tf
        if isinstance(data, np.ndarray):
            arr = tf.constant(data)
        else:
            arr = tf.constant(data)
        if dtype is not None:
            arr = tf.cast(arr, self._map_dtype(dtype))
        return arr

    # ---- 常用 dtype ----

    @property
    def int64(self):
        return self._tf.int64

    @property
    def float32(self):
        return self._tf.float32

    @property
    def float64(self):
        return self._tf.float64

    @property
    def bool(self):
        return self._tf.bool

    # ---- 数组操作 ----

    def where(self, condition, x, y):
        return self._tf.where(condition, x, y)

    def concatenate(self, arrays, axis=0):
        return self._tf.concat(list(arrays), axis=axis)

    def copy(self, arr):
        # TF tensor 是不可变的，copy 是恒等操作
        return arr

    def asarray(self, obj, dtype=None):
        tf = self._tf
        if isinstance(obj, tf.Tensor):
            arr = obj
        elif isinstance(obj, np.ndarray):
            arr = tf.constant(obj)
        else:
            arr = tf.constant(obj)
        if dtype is not None:
            arr = tf.cast(arr, self._map_dtype(dtype))
        return arr

    def ascontiguousarray(self, arr):
        # TF tensor 默认连续
        return arr

    # ---- 索引与查询 ----

    def nonzero(self, arr):
        tf = self._tf
        result = tf.where(arr)
        if result.shape.ndims > 1 and result.shape[-1] == 1:
            return tf.squeeze(result, axis=-1)
        return result

    def boolean_mask(self, arr, mask):
        return self._tf.boolean_mask(arr, mask)

    def isin(self, elements, test_elements):
        tf = self._tf
        if not isinstance(test_elements, tf.Tensor):
            test_elements = tf.constant(test_elements)
        if not isinstance(elements, tf.Tensor):
            elements = tf.constant(elements)
        # 使用 experimental numpy isin
        try:
            return tf.experimental.numpy.isin(elements, test_elements)
        except AttributeError:
            # 手动实现
            flat = tf.reshape(test_elements, [-1])
            expanded = tf.expand_dims(elements, -1)
            result = tf.reduce_any(tf.equal(expanded, flat), axis=-1)
            return result

    # ---- 聚合 ----

    def bincount(self, indices, weights=None, minlength=None):
        tf = self._tf
        if not isinstance(indices, tf.Tensor):
            indices = tf.constant(indices)
        indices = tf.cast(indices, tf.int32)
        ml = minlength if minlength is not None else 0
        if weights is not None:
            if not isinstance(weights, tf.Tensor):
                weights = tf.constant(weights)
            result = tf.math.bincount(indices, weights=weights, minlength=ml)
        else:
            result = tf.math.bincount(indices, minlength=ml)
        return result

    def add_at(self, output, indices, values):
        # TF tensor 是不可变的，add_at 返回新 tensor
        raise NotImplementedError(
            "TensorFlow tensor 是不可变的，add_at 无法原地修改。"
            "请使用 result = backend.add_at_return(output, indices, values) 代替。"
        )

    def add_at_return(self, output, indices, values):
        """TF 版本的 add_at，返回新 tensor 而非原地修改。"""
        tf = self._tf
        if not isinstance(indices, tf.Tensor):
            indices = tf.constant(indices)
        if not isinstance(values, tf.Tensor):
            values = tf.constant(values)
        indices = tf.expand_dims(indices, -1)
        return tf.tensor_scatter_nd_add(output, indices, values)

    def sum(self, arr, axis=None):
        tf = self._tf
        if axis is None:
            return tf.reduce_sum(arr)
        return tf.reduce_sum(arr, axis=axis)

    def all(self, arr, axis=None):
        tf = self._tf
        if axis is None:
            return tf.reduce_all(arr)
        return tf.reduce_all(arr, axis=axis)

    # ---- 排序与唯一 ----

    def argsort(self, arr, kind='stable'):
        tf = self._tf
        return tf.argsort(arr, stable=(kind == 'stable'))

    def flip(self, arr):
        tf = self._tf
        return tf.reverse(arr, axis=[0])

    def unique(self, arr):
        tf = self._tf
        result, _ = tf.unique(tf.reshape(arr, [-1]))
        return result

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
        return arr.numpy()

    def from_numpy(self, arr):
        return self._tf.constant(np.asarray(arr))

    # ---- 设备管理 ----

    def to_device(self, arr, device):
        tf = self._tf
        if device == 'cpu':
            with tf.device('/CPU:0'):
                return tf.identity(arr)
        elif device in ('gpu', 'cuda'):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                with tf.device('/GPU:0'):
                    return tf.identity(arr)
        return arr

    # ---- dtype 工具 ----

    def is_floating(self, dtype):
        tf = self._tf
        if isinstance(dtype, tf.DType):
            return dtype.is_floating
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, ValueError):
            return False

    def is_integer(self, dtype):
        tf = self._tf
        if isinstance(dtype, tf.DType):
            return dtype.is_integer
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, ValueError):
            return False

    def is_bool(self, dtype):
        tf = self._tf
        if isinstance(dtype, tf.DType):
            return dtype == tf.bool
        try:
            return np.issubdtype(dtype, np.bool_)
        except (TypeError, ValueError):
            return False

    def is_string(self, dtype):
        # TF 有 string dtype，但 RECS 不使用
        return False

    def dtype_of(self, arr):
        if hasattr(arr, 'dtype'):
            return arr.dtype
        return np.asarray(arr).dtype
