"""PyTorch 后端实现。

需要安装 recs[torch]（即 torch>=2.0）。
支持 CUDA / MPS 设备。
"""

import numpy as np

from .base import BackendBase


class TorchBackend(BackendBase):
    """PyTorch 后端：支持 CUDA / MPS GPU 加速。"""

    def __init__(self):
        import torch
        self._torch = torch
        self._device = 'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        )

    # ---- dtype 映射 ----

    _DTYPE_MAP = {
        'float32': None,  # 延迟初始化
        'float64': None,
        'int64': None,
        'bool': None,
    }

    def _init_dtype_map(self):
        if self._DTYPE_MAP['float32'] is None:
            t = self._torch
            self._DTYPE_MAP['float32'] = t.float32
            self._DTYPE_MAP['float64'] = t.float64
            self._DTYPE_MAP['int64'] = t.int64
            self._DTYPE_MAP['bool'] = t.bool

    def _map_dtype(self, dtype):
        """将 NumPy dtype 映射为 PyTorch dtype。"""
        self._init_dtype_map()
        if dtype is None:
            return None
        # 已经是 torch dtype
        if isinstance(dtype, self._torch.dtype):
            return dtype
        # NumPy dtype 或字符串
        dtype_str = str(np.dtype(dtype)) if not isinstance(dtype, str) else dtype
        mapping = {
            'float32': self._torch.float32,
            'float64': self._torch.float64,
            'int64': self._torch.int64,
            'int32': self._torch.int32,
            'int16': self._torch.int16,
            'int8': self._torch.int8,
            'bool': self._torch.bool,
            'uint8': self._torch.uint8,
        }
        mapped_dtype = mapping.get(dtype_str, self._torch.float32)
        
        # MPS 设备不支持 float64，自动降级为 float32
        if self._device == 'mps' and mapped_dtype == self._torch.float64:
            return self._torch.float32
        
        return mapped_dtype

    # ---- 元信息 ----

    @property
    def name(self) -> str:
        return 'torch'

    @property
    def device(self) -> str:
        return self._device

    # ---- 数组创建 ----

    def empty(self, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.empty(shape, dtype=self._map_dtype(dtype), device=self._device)

    def zeros(self, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.zeros(shape, dtype=self._map_dtype(dtype), device=self._device)
    def ones(self, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.ones(shape, dtype=self._map_dtype(dtype), device=self._device)
    def full(self, shape, fill_value, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.full(shape, fill_value, dtype=self._map_dtype(dtype), device=self._device)

    def arange(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            stop = start
            start = 0
        return self._torch.arange(start, stop, step, dtype=self._map_dtype(dtype), device=self._device)

    def array(self, data, dtype=None):
        t = self._torch
        if isinstance(data, np.ndarray):
            arr = t.from_numpy(data).to(self._device)
        else:
            arr = t.tensor(data, device=self._device)
        if dtype is not None:
            arr = arr.to(self._map_dtype(dtype))
        return arr

    # ---- 常用 dtype ----

    @property
    def int64(self):
        return self._torch.int64

    @property
    def float32(self):
        return self._torch.float32

    @property
    def float64(self):
        return self._torch.float64

    @property
    def bool(self):
        return self._torch.bool

    # ---- 数组操作 ----

    def where(self, condition, x, y):
        return self._torch.where(condition, x, y)

    def concatenate(self, arrays, axis=0):
        return self._torch.cat(list(arrays), dim=axis)

    def copy(self, arr):
        return arr.clone()

    def asarray(self, obj, dtype=None):
        t = self._torch
        if isinstance(obj, t.Tensor):
            arr = obj
        elif isinstance(obj, np.ndarray):
            arr = t.from_numpy(obj).to(self._device)
        else:
            arr = t.tensor(obj, device=self._device)
        if dtype is not None:
            arr = arr.to(self._map_dtype(dtype))
        return arr

    def ascontiguousarray(self, arr):
        return arr.contiguous()

    # ---- 索引与查询 ----

    def nonzero(self, arr):
        return self._torch.nonzero(arr, as_tuple=False).squeeze(-1)

    def boolean_mask(self, arr, mask):
        return arr[mask]

    def isin(self, elements, test_elements):
        t = self._torch
        if not isinstance(test_elements, t.Tensor):
            test_elements = t.tensor(test_elements, device=self._device)
        if not isinstance(elements, t.Tensor):
            elements = t.tensor(elements, device=self._device)
        # 手动实现 isin
        flat = test_elements.flatten()
        result = elements.flatten().unsqueeze(-1) == flat.unsqueeze(0)
        return result.any(dim=-1).reshape(elements.shape)

    # ---- 聚合 ----

    def bincount(self, indices, weights=None, minlength=None):
        t = self._torch
        if not isinstance(indices, t.Tensor):
            indices = t.tensor(indices, dtype=t.int64, device=self._device)
        indices = indices.long()
        ml = minlength if minlength is not None else 0
        if weights is not None:
            if not isinstance(weights, t.Tensor):
                weights = t.tensor(weights, device=self._device)
            result = t.bincount(indices, weights=weights, minlength=ml)
        else:
            result = t.bincount(indices, minlength=ml)
        # 确保长度至少为 minlength
        if ml > 0 and result.shape[0] < ml:
            pad = t.zeros(ml - result.shape[0], dtype=result.dtype, device=self._device)
            result = t.cat([result, pad])
        return result

    def add_at(self, output, indices, values):
        # PyTorch: index_add_ 是原地操作
        output.index_add_(0, indices.long(), values)

    def sum(self, arr, axis=None):
        if axis is None:
            return self._torch.sum(arr)
        return self._torch.sum(arr, dim=axis)

    def all(self, arr, axis=None):
        if axis is None:
            return self._torch.all(arr)
        return self._torch.all(arr, dim=axis)

    # ---- 排序与唯一 ----

    def argsort(self, arr, kind='stable'):
        return self._torch.argsort(arr, stable=(kind == 'stable'))

    def unique(self, arr):
        return self._torch.unique(arr)

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
        # MPS/CUDA 设备上的张量需要先移到 CPU
        if arr.device.type != 'cpu':
            arr = arr.cpu()
        return arr.detach().numpy()

    def from_numpy(self, arr):
        return self._torch.from_numpy(np.asarray(arr)).to(self._device)

    # ---- 设备管理 ----

    def to_device(self, arr, device):
        if device == 'gpu':
            device = self._device
        return arr.to(device)

    # ---- dtype 工具 ----

    def is_floating(self, dtype):
        t = self._torch
        if isinstance(dtype, t.dtype):
            return dtype in (t.float16, t.float32, t.float64, t.bfloat16)
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, ValueError):
            return False

    def is_integer(self, dtype):
        t = self._torch
        if isinstance(dtype, t.dtype):
            return dtype in (t.int8, t.int16, t.int32, t.int64, t.uint8)
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, ValueError):
            return False

    def is_bool(self, dtype):
        t = self._torch
        if isinstance(dtype, t.dtype):
            return dtype == t.bool
        try:
            return np.issubdtype(dtype, np.bool_)
        except (TypeError, ValueError):
            return False

    def is_string(self, dtype):
        # PyTorch 不支持字符串张量
        return False

    def dtype_of(self, arr):
        if isinstance(arr, self._torch.Tensor):
            return arr.dtype
        return np.asarray(arr).dtype
