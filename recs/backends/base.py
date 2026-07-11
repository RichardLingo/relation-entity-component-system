"""RECS 计算后端抽象基类。

定义 EntityPool 和 Relation 所需的全部数组操作原语。
每个具体后端实现约 25 个方法即可接入 RECS 多后端体系。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union


class BackendBase(ABC):
    """RECS 计算后端抽象。

    所有后端必须实现以下方法，以保证 EntityPool / Relation 的核心逻辑
    可以在不同数组框架（NumPy / PyTorch / MLX / JAX / TF）上透明运行。
    """

    # ---- 元信息 ----

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称：'numpy' | 'torch' | 'mlx' | 'jax' | 'tensorflow'。"""

    @property
    @abstractmethod
    def device(self) -> str:
        """当前默认设备：'cpu' | 'cuda' | 'mps' | 'gpu'。"""

    # ---- 数组创建 ----

    @abstractmethod
    def empty(self, shape: Union[int, tuple], dtype: Any) -> Any:
        """创建未初始化的数组。"""

    @abstractmethod
    def zeros(self, shape: Union[int, tuple], dtype: Any) -> Any:
        """创建全零数组。"""

    @abstractmethod
    def ones(self, shape: Union[int, tuple], dtype: Any) -> Any:
        """创建全 1 数组。"""

    @abstractmethod
    def full(self, shape: Union[int, tuple], fill_value: Any, dtype: Any) -> Any:
        """创建填充指定值的数组。"""

    @abstractmethod
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None) -> Any:
        """创建等差数组（类 np.arange）。"""

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        """从 Python 列表/标量创建数组。"""

    # ---- 常用 dtype ----

    @property
    @abstractmethod
    def int64(self) -> Any:
        """int64 类型。"""

    @property
    @abstractmethod
    def float32(self) -> Any:
        """float32 类型。"""

    @property
    @abstractmethod
    def float64(self) -> Any:
        """float64 类型。"""

    @property
    @abstractmethod
    def bool(self) -> Any:
        """bool 类型。"""

    # ---- 数组操作 ----

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """条件选择：condition 为真取 x，否则取 y。"""

    @abstractmethod
    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        """沿指定轴拼接数组。"""

    @abstractmethod
    def copy(self, arr: Any) -> Any:
        """返回数组的深拷贝。"""

    @abstractmethod
    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        """将输入转为数组（若已是目标类型则避免拷贝）。"""

    @abstractmethod
    def ascontiguousarray(self, arr: Any) -> Any:
        """确保数组在内存中连续存储（C-order）。"""

    # ---- 索引与查询 ----

    @abstractmethod
    def nonzero(self, arr: Any) -> Any:
        """返回非零元素的索引。"""

    @abstractmethod
    def boolean_mask(self, arr: Any, mask: Any) -> Any:
        """按布尔掩码筛选元素。"""

    @abstractmethod
    def isin(self, elements: Any, test_elements: Any) -> Any:
        """判断 elements 中每个元素是否在 test_elements 中。"""

    # ---- 聚合 ----

    @abstractmethod
    def bincount(self, indices: Any, weights: Any = None, minlength: Optional[int] = None) -> Any:
        """按索引计数（可选加权）。"""

    @abstractmethod
    def add_at(self, output: Any, indices: Any, values: Any) -> None:
        """向量化 scatter add：output[indices] += values（原地修改）。"""

    @abstractmethod
    def sum(self, arr: Any, axis: Optional[int] = None) -> Any:
        """求和。"""

    @abstractmethod
    def all(self, arr: Any, axis: Optional[int] = None) -> Any:
        """判断所有元素是否为真。"""

    # ---- 排序与唯一 ----

    @abstractmethod
    def argsort(self, arr: Any, kind: str = 'stable') -> Any:
        """返回排序索引。"""

    @abstractmethod
    def flip(self, arr: Any) -> Any:
        """反转数组元素顺序。"""

    @abstractmethod
    def unique(self, arr: Any) -> Any:
        """返回去重后的唯一元素。"""

    # ---- 位操作（SparseBoolBitset 专用） ----

    @abstractmethod
    def packbits(self, arr: Any, bitorder: str = 'little') -> Any:
        """将布尔数组压缩为位数组。"""

    @abstractmethod
    def unpackbits(self, arr: Any, count: int, bitorder: str = 'little') -> Any:
        """将位数组解压缩为布尔数组，截取前 count 个元素。"""

    # ---- 类型转换 ----

    @abstractmethod
    def to_numpy(self, arr: Any) -> "np.ndarray":
        """将后端数组转为 NumPy ndarray。"""

    @abstractmethod
    def from_numpy(self, arr: "np.ndarray") -> Any:
        """将 NumPy ndarray 转为后端数组。"""

    # ---- 设备管理 ----

    @abstractmethod
    def to_device(self, arr: Any, device: str) -> Any:
        """将数组移动到指定设备。"""

    def cpu(self, arr: Any) -> Any:
        """将数组移到 CPU。"""
        return self.to_device(arr, 'cpu')

    def gpu(self, arr: Any) -> Any:
        """将数组移到 GPU（具体设备由后端决定）。"""
        return self.to_device(arr, 'gpu')

    # ---- 类型转换工具 ----

    def astype(self, arr: Any, dtype: Any) -> Any:
        """转换数组的 dtype。默认实现委托给标准类型转换。"""
        return self.asarray(arr, dtype=dtype)

    # ---- dtype 工具 ----

    @abstractmethod
    def is_floating(self, dtype: Any) -> bool:
        """判断 dtype 是否为浮点类型。"""

    @abstractmethod
    def is_integer(self, dtype: Any) -> bool:
        """判断 dtype 是否为整数类型。"""

    @abstractmethod
    def is_bool(self, dtype: Any) -> bool:
        """判断 dtype 是否为布尔类型。"""

    @abstractmethod
    def is_string(self, dtype: Any) -> bool:
        """判断 dtype 是否为字符串类型。"""

    @abstractmethod
    def dtype_of(self, arr: Any) -> Any:
        """返回数组的 dtype。"""
