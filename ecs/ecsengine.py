import numpy as np
from typing import Optional

# 后端支持：延迟导入以避免循环依赖
def _get_backend(backend_name: str = 'numpy'):
    """获取后端实例。"""
    from recs.backends import get_backend
    return get_backend(backend_name)


def _is_string_dtype(dtype) -> bool:
    """检测 dtype 是否为字符串类型。

    字符串列必须强制使用 NumPy 后端，因为 PyTorch/MLX/JAX/TF 不支持字符串数组。
    支持 NumPy、PyTorch、MLX、JAX、TensorFlow 等后端的类型检测。
    """
    if dtype is None:
        return False
    
    # 检查是否为 Python str 类型
    if dtype == str:
        return True
    
    # 尝试转换为 NumPy dtype
    try:
        dtype_str = str(np.dtype(dtype))
        return dtype_str.startswith('<U') or dtype_str.startswith('>U') or dtype_str == 'object'
    except (TypeError, AttributeError):
        # 如果不是 NumPy dtype，检查是否为字符串格式
        dtype_str = str(dtype)
        return dtype_str.startswith('<U') or dtype_str.startswith('>U') or 'str' in dtype_str.lower()


def _is_string_value(value) -> bool:
    """检测值是否为字符串或包含字符串的可迭代对象。

    用于在后端转换前提前判断，避免 PyTorch/MLX 等后端处理字符串时报错。
    """
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(v, str) for v in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in ('U', 'S', 'O')
    return False


# =============================
# 通用：索引规范化（类 NumPy）
# =============================

# 说明：
# - idx 支持：None/int/slice/int-array/bool-mask
# - 统一返回 1D int indices 或 1D bool mask


def to_indices(idx, size: int) -> np.ndarray:
    """将 NumPy 风格索引规范化为一维整数 indices。"""
    if idx is None:
        return np.arange(size, dtype=int)

    if isinstance(idx, (int, np.integer)):
        i = int(idx)
        if i < 0:
            i += size
        if i < 0 or i >= size:
            raise IndexError("索引越界")
        return np.array([i], dtype=int)

    if isinstance(idx, slice):
        start, stop, step = idx.indices(size)
        return np.arange(start, stop, step, dtype=int)

    arr = np.asarray(idx)
    if arr.dtype == np.bool_:
        if arr.ndim != 1 or arr.shape[0] != size:
            raise ValueError(f"布尔掩码长度必须等于 size ({size})")
        return np.nonzero(arr)[0]

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(int, copy=False)
        neg = arr < 0
        if np.any(neg):
            arr = arr.copy()
            arr[neg] += size
        if np.any((arr < 0) | (arr >= size)):
            raise IndexError("索引越界")
        return arr

    raise TypeError(f"不支持的索引类型: {type(idx)}")


def to_mask(sel, size: int) -> np.ndarray:
    """将选择器规范化为长度为 size 的布尔掩码。"""
    if sel is None:
        return np.ones(size, dtype=bool)

    if isinstance(sel, np.ndarray) and sel.dtype == np.bool_:
        if sel.ndim != 1 or sel.shape[0] != size:
            raise ValueError(f"布尔掩码长度必须等于 size ({size})")
        return sel.astype(bool, copy=False)

    idxs = to_indices(sel, size)
    mask = np.zeros(size, dtype=bool)
    if idxs.size:
        mask[idxs] = True
    return mask


def _normalize_backend(backend: str) -> str:
    """规范化后端参数并做合法性校验。

    Args:
        backend: 后端类型，允许 ``'auto'``、``'dense'``、``'sparse'``。

    Returns:
        规范化后的后端字符串。

    Raises:
        ValueError: 当 backend 不在允许集合内时抛出。
    """
    b = str(backend)
    if b not in ('auto', 'dense', 'sparse'):
        raise ValueError("backend 必须为 'auto'/'dense'/'sparse'")
    return b


# =============================
# 第一类二维数组：实体表（稀疏列）
# =============================


def _packbits(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("mask 必须为一维")
    return np.packbits(mask, bitorder="little")


def _unpackbits(packed: np.ndarray, n: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.unpackbits(packed, bitorder="little")
    return out[:n].astype(bool, copy=False)


class SparseBoolBitset:
    """稀疏布尔列：使用 packbits 压缩存储。"""

    def __init__(self, packed: np.ndarray, n: int):
        self.packed = np.asarray(packed, dtype=np.uint8)
        self.n = int(n)

    @classmethod
    def from_dense(cls, mask) -> "SparseBoolBitset":
        dense = np.asarray(mask, dtype=bool)
        return cls(_packbits(dense), int(dense.shape[0]))

    def to_dense(self) -> np.ndarray:
        return _unpackbits(self.packed, self.n)

    def where_true(self) -> np.ndarray:
        return np.nonzero(self.to_dense())[0]

    def take(self, idx) -> "SparseBoolBitset":
        dense = self.to_dense()
        idxs = to_indices(idx, self.n)
        return SparseBoolBitset.from_dense(dense[idxs])


class SparseNullable:
    """稀疏缺失值列：present 位图 + values_idx + values。"""

    def __init__(self, present: SparseBoolBitset, values_idx: np.ndarray, values: np.ndarray):
        self.present = present
        self.values_idx = np.asarray(values_idx, dtype=int)
        self.values = np.asarray(values)

    @classmethod
    def from_dense(cls, dense: np.ndarray, missing_value=None) -> "SparseNullable":
        arr = np.asarray(dense)
        if missing_value is None:
            present = arr != None  # noqa: E711
        else:
            present = arr != missing_value
        idx = np.nonzero(present)[0].astype(int)
        vals = arr[idx]
        return cls(SparseBoolBitset.from_dense(present), idx, vals)

    @property
    def n(self) -> int:
        return int(self.present.n)

    def present_mask(self) -> np.ndarray:
        return self.present.to_dense()

    def take(self, idx) -> "SparseNullable":
        idxs = to_indices(idx, self.n)
        old_to_new = np.full(self.n, -1, dtype=int)
        old_to_new[idxs] = np.arange(idxs.size, dtype=int)
        in_sel = old_to_new[self.values_idx] >= 0
        new_values_idx = old_to_new[self.values_idx[in_sel]]
        new_values = self.values[in_sel]
        new_present = np.zeros(idxs.size, dtype=bool)
        if new_values_idx.size:
            new_present[new_values_idx] = True
        return SparseNullable(SparseBoolBitset.from_dense(new_present), new_values_idx.astype(int, copy=False), new_values)


# =============================
# 第二类二维数组：Relation（RECS 的关系扩展，稠密/稀疏后端）
# =============================


class DenseRelation:
    """RECS 稠密关系：用二维 ndarray 表示。"""

    def __init__(self, mat: np.ndarray):
        arr = np.asarray(mat)
        if arr.ndim != 2:
            raise ValueError("mat 必须是二维数组")
        self.mat = arr

    @property
    def shape(self):
        return self.mat.shape

    def index(self, rows=None, cols=None, *, return_: str = 'indices'):
        """索引层：类 NumPy 的 mat[rows, cols] 选择。

        return_:
        - 'indices': 返回 (row_indices, col_indices)
        - 'mask': 返回子矩阵掩码（同 shape）
        """
        n_rows, n_cols = self.mat.shape
        r = to_indices(rows, n_rows)
        c = to_indices(cols, n_cols)
        if return_ == 'indices':
            return r, c
        if return_ == 'mask':
            m = np.zeros(self.mat.shape, dtype=bool)
            m[np.ix_(r, c)] = True
            return m
        raise ValueError("return_ 必须为 'indices' 或 'mask'")

    def query(self, rows=None, cols=None, *, pred=None, return_: str = 'dense'):
        """查询层：返回子矩阵或坐标。"""
        sub = self.mat
        if rows is not None or cols is not None:
            n_rows, n_cols = self.mat.shape
            r = to_indices(rows, n_rows)
            c = to_indices(cols, n_cols)
            sub = self.mat[np.ix_(r, c)]
        if pred is None:
            if return_ == 'dense':
                return sub
            raise ValueError("pred 为 None 时仅支持 return_='dense'")
        mask = pred(sub) if callable(pred) else pred
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != sub.shape:
            raise ValueError("谓词掩码形状不匹配")
        rows2, cols2 = np.nonzero(mask)
        if return_ == 'coords':
            return rows2, cols2
        if return_ == 'indices':
            return rows2 * sub.shape[1] + cols2
        raise ValueError("return_ 必须为 'dense'/'coords'/'indices'")


class SparseRelationEdges:
    """RECS 稀疏关系：边表（src_uid/dst_uid/value）。

    侧重查询/索引（不提供完整增删 API）。
    """

    def __init__(self, src_uid: np.ndarray, dst_uid: np.ndarray, value: Optional[np.ndarray] = None):
        self.src_uid = np.asarray(src_uid, dtype=np.int64)
        self.dst_uid = np.asarray(dst_uid, dtype=np.int64)
        if self.src_uid.shape != self.dst_uid.shape:
            raise ValueError("src_uid 与 dst_uid 长度必须相等")
        self.value = None if value is None else np.asarray(value)
        if self.value is not None and self.value.shape != self.src_uid.shape:
            raise ValueError("value 长度必须与 src_uid 相等")

    @property
    def size(self) -> int:
        return int(self.src_uid.size)

    @staticmethod
    def build_uid_to_pos(uid_array: np.ndarray) -> np.ndarray:
        uids = np.asarray(uid_array, dtype=np.int64)
        if uids.size == 0:
            return np.empty(0, dtype=int)
        max_uid = int(uids.max())
        uid_to_pos = np.full(max_uid + 1, -1, dtype=int)
        uid_to_pos[uids] = np.arange(uids.size, dtype=int)
        return uid_to_pos

    @staticmethod
    def map_uids_to_pos(uids: np.ndarray, uid_to_pos) -> np.ndarray:
        arr = np.asarray(uid_to_pos)
        uids = np.asarray(uids, dtype=np.int64)
        out = np.full(uids.shape, -1, dtype=int)
        valid = (uids >= 0) & (uids < arr.shape[0])
        if np.any(valid):
            out[valid] = arr[uids[valid]]
        return out

    def index(self, *, src_uid=None, dst_uid=None, edge_pred=None, return_: str = 'indices'):
        """索引层：返回边索引或 mask。"""
        E = self.size
        mask = np.ones(E, dtype=bool)
        if src_uid is not None:
            if isinstance(src_uid, (int, np.integer)):
                mask &= (self.src_uid == int(src_uid))
            else:
                arr = np.asarray(src_uid, dtype=np.int64)
                mask &= np.isin(self.src_uid, arr)
        if dst_uid is not None:
            if isinstance(dst_uid, (int, np.integer)):
                mask &= (self.dst_uid == int(dst_uid))
            else:
                arr = np.asarray(dst_uid, dtype=np.int64)
                mask &= np.isin(self.dst_uid, arr)
        if edge_pred is not None:
            em = edge_pred(self) if callable(edge_pred) else edge_pred
            em = np.asarray(em, dtype=bool)
            if em.shape[0] != E:
                raise ValueError("edge_pred 掩码长度必须等于边数")
            mask &= em
        return mask if return_ == 'mask' else np.nonzero(mask)[0]


class Relation:
    """
    RECS 边表关系（SoA 风格）：以 src_uid / dst_uid 为主键存储，其他属性按列保存。

    设计原则：
    - 存储 uid 而非 pool 引用，以便实体可以移动/压缩而关系保持稳定。
    - 提供批量添加、按 src/dst 查询，以及能从稠密矩阵构造等功能。
    - 支持多后端计算（NumPy/PyTorch/MLX/JAX/TensorFlow）
    """

    def __init__(self, name: str, capacity: int = 1024, attr_dtypes: Optional[dict] = None, backend: str = 'numpy'):
        self.name = name
        self.capacity = int(capacity)
        self.size = 0
        self._capacity = self.capacity
        
        # 多后端支持
        self.backend = _get_backend(backend)
        self._string_columns = set()
        
        self.d = {}
        self.d['src_uid'] = self.backend.empty(self.capacity, dtype=np.int64)
        self.d['dst_uid'] = self.backend.empty(self.capacity, dtype=np.int64)
        if attr_dtypes:
            for k, dt in attr_dtypes.items():
                if k in ('src_uid', 'dst_uid'):
                    continue
                # 字符串列强制使用 NumPy
                if _is_string_dtype(dt):
                    self.d[k] = np.empty(self.capacity, dtype=dt)
                    self._string_columns.add(k)
                else:
                    self.d[k] = self.backend.empty(self.capacity, dtype=dt)

    def _resize(self, new_cap: int):
        if new_cap <= self._capacity:
            return
        old = self._capacity
        for k, arr in list(self.d.items()):
            dtype = arr.dtype
            # 字符串列强制使用 NumPy
            if k in self._string_columns:
                new_arr = np.empty(new_cap, dtype=dtype)
                # 如果原数组是 backend 数组，需要转换
                if not isinstance(arr, np.ndarray):
                    new_arr[:self.size] = self.backend.to_numpy(arr[:self.size])
                else:
                    new_arr[:self.size] = arr[:self.size]
            else:
                new_arr = self.backend.empty(new_cap, dtype=dtype)
                new_arr[:self.size] = arr[:self.size]
            # 尾部初始化
            if self.backend.is_floating(dtype) or self.backend.is_integer(dtype) or self.backend.is_bool(dtype):
                new_arr[old:new_cap] = 0
            else:
                new_arr[old:new_cap] = None
            self.d[k] = new_arr
        self._capacity = new_cap

    def add(self, src_uids, dst_uids, **attrs):
        src_u = self.backend.asarray(src_uids, dtype=np.int64)
        dst_u = self.backend.asarray(dst_uids, dtype=np.int64)
        if src_u.shape != dst_u.shape:
            raise ValueError("src and dst must have same shape")
        n = src_u.shape[0]
        if n == 0:
            return self.backend.empty(0, dtype=int)
        if self.size + n > self._capacity:
            self._resize(max(self._capacity * 2, self.size + n))
        idxs = self.backend.arange(self.size, self.size + n, dtype=int)
        self._set_item_with_immutable_check('src_uid', idxs, src_u)
        self._set_item_with_immutable_check('dst_uid', idxs, dst_u)
        for k, v in attrs.items():
            if k not in self.d:
                # 字符串值强制使用 NumPy 检测
                if _is_string_value(v):
                    arr = np.empty(self._capacity, dtype='U10')
                    self._string_columns.add(k)
                    dtype = arr.dtype
                else:
                    dtype = self.backend.dtype_of(self.backend.asarray(v))
                    # 字符串列强制使用 NumPy
                    if _is_string_dtype(dtype):
                        arr = np.empty(self._capacity, dtype=dtype)
                        self._string_columns.add(k)
                    else:
                        arr = self.backend.empty(self._capacity, dtype=dtype)
                # 初始化尾部
                if self.backend.is_floating(dtype) or self.backend.is_integer(dtype) or self.backend.is_bool(dtype):
                    arr[self.size:self._capacity] = 0
                else:
                    arr[self.size:self._capacity] = None
                self.d[k] = arr
            if _is_string_value(v) or k in self._string_columns:
                val = np.asarray(v)
                # 字符串列是 NumPy，使用 NumPy 索引
                idxs_np = self.backend.to_numpy(idxs)
                if val.shape[0] == 1:
                    self.d[k][idxs_np] = val.item()
                else:
                    if val.shape[0] != n:
                        raise ValueError(f"Length mismatch for edge attr {k}")
                    self.d[k][idxs_np] = val
            else:
                val = self.backend.asarray(v)
                self._set_item_with_immutable_check(k, idxs, val)
        self.size += n
        return idxs

    def remove_by_indices(self, idxs):
        idxs = self.backend.unique(self.backend.asarray(idxs, dtype=int))
        valid = (idxs >= 0) & (idxs < self.size)
        if not self.backend.all(valid):
            raise IndexError("Index out of bounds")
        mask = self.backend.ones(self.size, dtype=bool)
        # 使用 where 来创建反向掩码，而不是直接修改
        idxs_np = self.backend.to_numpy(idxs)
        mask_np = np.ones(self.size, dtype=bool)
        mask_np[idxs_np] = False
        mask = self.backend.array(mask_np)
        new_size = int(self.backend.sum(mask))
        # 注意：self.d[k] 是 capacity 大小，mask 是 self.size 大小，必须切片
        for k in self.d:
            if k in self._string_columns:
                # 字符串列使用 NumPy 处理
                self.d[k][:new_size] = self.d[k][:self.size][mask_np]
            else:
                # 使用 boolean_mask 处理，避免直接布尔索引
                selected = self.backend.boolean_mask(self.d[k][:self.size], mask)
                # 对于不可变数组，需要用 concatenate 重新构造整个数组
                zeros_tail = self.backend.zeros(self._capacity - new_size, dtype=self.d[k].dtype)
                self.d[k] = self.backend.concatenate([selected, zeros_tail])
        self.size = new_size

    def neighbors_from_uid(self, src_uid):
        mask = self.d['src_uid'][:self.size] == src_uid
        idxs = self.backend.nonzero(mask)
        dsts = self.d['dst_uid'][idxs]
        attrs = {k: v[idxs] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        return dsts, attrs

    def neighbors_to_uid(self, dst_uid):
        mask = self.d['dst_uid'][:self.size] == dst_uid
        idxs = self.backend.nonzero(mask)
        srcs = self.d['src_uid'][idxs]
        attrs = {k: v[idxs] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        return srcs, attrs

    def add_attribute(self, name, dtype, default=0):
        if name in self.d:
            raise ValueError(f"Attribute '{name}' already exists.")
        # 字符串列强制使用 NumPy
        if _is_string_dtype(dtype):
            self._string_columns.add(name)
            try:
                arr = np.full(self._capacity, default, dtype=dtype)
            except Exception:
                arr = np.empty(self._capacity, dtype=dtype)
                arr[:self.size] = default
                if self.size < self._capacity:
                    arr[self.size:self._capacity] = default
        else:
            try:
                arr = self.backend.full(self._capacity, default, dtype=dtype)
            except Exception:
                arr = self.backend.empty(self._capacity, dtype=dtype)
                arr[:self.size] = default
                if self.size < self._capacity:
                    arr[self.size:self._capacity] = default
        self.d[name] = arr

    def get_attr(self, name):
        return self.d[name][:self.size]

    def get_attrs_view(self, names):
        return {name: self.get_attr(name) for name in names}

    def _set_item_with_immutable_check(self, name, idxs, values):
        """处理不可变数组的赋值（如 JAX）。"""
        arr = self.d[name]
        try:
            arr[idxs] = values
        except TypeError as e:
            # 可能是不可变数组（如 JAX）
            if 'immutable' in str(e).lower() or 'not support item assignment' in str(e).lower():
                # 使用 .at[idx].set(y) 模式
                if hasattr(arr, 'at'):
                    self.d[name] = arr.at[idxs].set(values)
                else:
                    raise
            else:
                raise

    def _to_indices(self, idx):
        """
        将多种索引输入统一为整型索引数组。
        支持：int、slice、布尔掩码、整型数组/可迭代。
        """
        # 单个整数
        if isinstance(idx, (int, np.integer)):
            return self.backend.array([int(idx)], dtype=int)

        # slice
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            return self.backend.arange(start, stop, step, dtype=int)

        arr = self.backend.asarray(idx)
        # 布尔掩码
        if self.backend.is_bool(arr.dtype):
            if arr.shape[0] != self.size:
                raise ValueError(f"Boolean mask length must equal current size ({self.size})")
            return self.backend.nonzero(arr)

        # 整型数组或可迭代
        if self.backend.is_integer(arr.dtype):
            return self.backend.astype(arr, int)

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def indices_from_uid(self, src_uid):
        return self.backend.nonzero(self.d['src_uid'][:self.size] == src_uid)

    def indices_to_uid(self, dst_uid):
        return self.backend.nonzero(self.d['dst_uid'][:self.size] == dst_uid)

    def take(self, idx):
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return Relation(self.name, capacity=4, backend=self.backend.name)
        attr_dtypes = {k: v.dtype for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        rel = Relation(self.name, capacity=max(8, idxs.size), attr_dtypes=attr_dtypes, backend=self.backend.name)
        src_u = self.d['src_uid'][idxs]
        dst_u = self.d['dst_uid'][idxs]
        attrs = {k: v[idxs] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        rel.add(src_u, dst_u, **attrs)
        return rel

    def remove_by_mask(self, mask):
        mask = self.backend.asarray(mask, dtype=bool)
        if mask.shape[0] != self.size:
            raise ValueError(f"Mask length must equal current size ({self.size})")
        keep = ~mask
        keep_count = int(self.backend.sum(keep))
        keep_np = self.backend.to_numpy(keep)
        for k in self.d:
            if k in self._string_columns:
                # 字符串列是 NumPy，使用 NumPy 掩码
                self.d[k][:keep_count] = self.d[k][:self.size][keep_np]
            else:
                # 数值列使用 backend 的 boolean_mask
                selected = self.backend.boolean_mask(self.d[k][:self.size], keep)
                # 对于不可变数组，需要用 concatenate 重新构造整个数组
                zeros_tail = self.backend.zeros(self._capacity - keep_count, dtype=self.d[k].dtype)
                self.d[k] = self.backend.concatenate([selected, zeros_tail])
        self.size = keep_count

    def set_attr(self, name, idx, values, *, backend: str = 'auto'):
        """按索引对单列执行赋值（类 NumPy 语义）。

        Args:
            name: 列名。
            idx: 目标边索引，支持 int/slice/布尔掩码/整型数组。
            values: 标量或与索引长度一致的一维数组。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。

        Raises:
            KeyError: 列不存在。
            ValueError: 向量赋值长度与索引长度不一致。
        """
        _normalize_backend(backend)
        if name not in self.d:
            raise KeyError(f"Attribute '{name}' not found")
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return
        val = self.backend.asarray(values)
        if val.size == 1:
            self.d[name][idxs] = val.item()
            return
        if val.shape[0] != idxs.size:
            raise ValueError(f"Length mismatch for attribute '{name}': expected {idxs.size}, got {val.shape[0]}")
        self.d[name][idxs] = val

    def assign(self, idx, values_by_attr: dict, *, backend: str = 'auto'):
        """按同一索引对多列执行批量赋值。

        Args:
            idx: 目标边索引，支持 int/slice/布尔掩码/整型数组。
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        _normalize_backend(backend)
        if not isinstance(values_by_attr, dict) or len(values_by_attr) == 0:
            return
        for name, values in values_by_attr.items():
            self.set_attr(name, idx, values, backend=backend)

    def where_set(
            self,
            *,
            src_uid: Optional[object] = None,
            dst_uid: Optional[object] = None,
            edge_pred=None,
            values_by_attr: dict,
            backend: str = 'auto'
    ):
        """按查询条件对多列执行批量赋值。

        Args:
            src_uid: 源实体 uid 条件。
            dst_uid: 目标实体 uid 条件。
            edge_pred: 边谓词条件，可为布尔掩码或 ``callable(rel) -> mask``。
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。

        Returns:
            命中的边索引数组。
        """
        _normalize_backend(backend)
        idxs = self.query(src_uid=src_uid, dst_uid=dst_uid, edge_pred=edge_pred, return_='indices')
        self.assign(idxs, values_by_attr, backend=backend)
        return idxs

    def argsort_by(self, name, ascending: bool = True):
        arr = self.get_attr(name)
        order = self.backend.argsort(arr, kind='stable')
        return order if ascending else order[::-1]

    def sort_by(self, name, ascending: bool = True):
        order = self.argsort_by(name, ascending=ascending)
        order_np = self.backend.to_numpy(order)
        for k in self.d:
            if k in self._string_columns:
                # 字符串列是 NumPy，使用 NumPy 索引
                self.d[k][:self.size] = self.d[k][:self.size][order_np]
            else:
                # 数值列使用 backend 的 take
                self.d[k][:self.size] = self.backend.take(self.d[k][:self.size], order)
        return order

    @staticmethod
    def build_uid_to_pos(uid_array) -> np.ndarray:
        """构建 uid 到位置的映射数组（静态方法，始终使用 NumPy）。"""
        uids = np.asarray(uid_array, dtype=np.int64)
        if uids.size == 0:
            return np.empty(0, dtype=int)
        max_uid = int(uids.max())
        uid_to_pos = np.full(max_uid + 1, -1, dtype=int)
        uid_to_pos[uids] = np.arange(uids.size, dtype=int)
        return uid_to_pos

    @staticmethod
    def _map_uids_to_pos(uids: np.ndarray, uid_to_pos):
        """将 uid 数组映射为位置数组（静态方法，始终使用 NumPy）。"""
        if isinstance(uid_to_pos, dict):
            return np.fromiter((uid_to_pos.get(int(u), -1) for u in uids), dtype=int, count=uids.size)
        arr = np.asarray(uid_to_pos)
        out = np.full(uids.shape, -1, dtype=int)
        valid = (uids >= 0) & (uids < arr.shape[0])
        if np.any(valid):
            out[valid] = arr[uids[valid]]
        return out

    def _bincount(self, uid_field: str, attr_name: Optional[str], uid_to_pos, n_out: Optional[int]):
        uids = self.d[uid_field][:self.size]
        pos = self._map_uids_to_pos(uids, uid_to_pos)
        valid = pos >= 0
        if n_out is None:
            n_out = int(pos[valid].max()) + 1 if np.any(valid) else 0
        weights = None
        if attr_name is not None:
            weights = self.d[attr_name][:self.size][valid]
        return np.bincount(pos[valid], weights=weights, minlength=int(n_out))

    def row_sum(self, attr_name: str, src_uid_to_pos, n_rows: Optional[int] = None):
        return self._bincount('src_uid', attr_name, src_uid_to_pos, n_rows)

    def col_sum(self, attr_name: str, dst_uid_to_pos, n_cols: Optional[int] = None):
        return self._bincount('dst_uid', attr_name, dst_uid_to_pos, n_cols)

    def row_count(self, src_uid_to_pos, n_rows: Optional[int] = None):
        return self._bincount('src_uid', None, src_uid_to_pos, n_rows)

    def col_count(self, dst_uid_to_pos, n_cols: Optional[int] = None):
        return self._bincount('dst_uid', None, dst_uid_to_pos, n_cols)

    def query(
            self,
            *,
            src_uid: Optional[object] = None,
            dst_uid: Optional[object] = None,
            edge_pred=None,
            return_: str = 'indices'
    ):
        """高阶查询：根据条件返回边集合（indices/mask/子 Relation）。

        这是“查询层”API：它可以组合多种条件，内部会构造 mask，然后按 return_ 返回结果。

        参数
        ----
        src_uid, dst_uid:
            低阶索引条件（按 uid 过滤）。

            - None：不约束该端
            - 单个 uid（int / np.integer）
            - 多个 uid（序列/数组）：语义为 OR，即“src_uid 属于给定集合”

        edge_pred:
            边属性谓词。可为：
            - None
            - 布尔掩码（长度 == self.size）
            - callable(rel) -> 布尔掩码

        return_:
            - 'indices': 返回满足条件的边索引（np.ndarray[int]）
            - 'mask': 返回满足条件的布尔掩码（np.ndarray[bool]，len==size）
            - 'relation': 返回子 Relation（注意：这是拷贝，会分配新数组）
        """
        size = self.size
        if size == 0:
            if return_ == 'mask':
                return np.zeros(0, dtype=bool)
            if return_ == 'indices':
                return np.empty(0, dtype=int)
            return Relation(self.name, capacity=4)

        mask = np.ones(size, dtype=bool)

        def _uid_or_mask(field: str, u):
            if u is None:
                return None
            # 单个 uid
            if isinstance(u, (int, np.integer)):
                return self.d[field][:size] == int(u)
            # 多个 uid：OR
            arr = np.asarray(u, dtype=np.int64)
            # 空集合：直接无结果
            if arr.size == 0:
                return np.zeros(size, dtype=bool)
            return np.isin(self.d[field][:size], arr)

        # uid 条件（索引层）
        m = _uid_or_mask('src_uid', src_uid)
        if m is not None:
            mask &= m
        m = _uid_or_mask('dst_uid', dst_uid)
        if m is not None:
            mask &= m

        # 边属性谓词（查询层）
        if edge_pred is not None:
            em = edge_pred(self) if callable(edge_pred) else edge_pred
            em = np.asarray(em, dtype=bool)
            if em.shape[0] != size:
                raise ValueError(f"edge_pred 掩码长度必须等于 relation.size ({size})")
            mask &= em

        if return_ == 'mask':
            return mask

        idxs = np.nonzero(mask)[0]
        if return_ == 'indices':
            return idxs

        if return_ == 'view':
            return RelationView(self, idxs)

        if return_ == 'relation':
            return self.take(idxs)

        raise ValueError("return_ 必须为 'indices'/'mask'/'view'/'relation'")

    def index(
            self,
            *,
            src_uid: Optional[object] = None,
            dst_uid: Optional[object] = None,
            edge_pred=None,
            return_: str = 'indices'
    ):
        """索引层：只返回边的位置（indices/mask）。

        这是 `Relation.query(...)` 的轻量子集，便于用户组合多个条件：

        - 返回 indices/mask，不材料化子 Relation。
        - 参数语义与 query 保持一致。
        """
        if return_ not in ('indices', 'mask'):
            raise ValueError("Relation.index 的 return_ 只支持 'indices' 或 'mask'")
        return self.query(src_uid=src_uid, dst_uid=dst_uid, edge_pred=edge_pred, return_=return_)


class RelationView:
    """Relation 的轻量视图：只保存 (relation, edge_indices)。

    用途
    ----
    - 避免 `return_='relation'` 的材料化拷贝。
    - 允许像“表”一样取列：view['amount'] / view.amount。

    注意
    ----
    - 对列做 fancy indexing 通常仍会产生拷贝。
      view 的优势在于：延迟取列、只取需要的列、idx 可复用组合。
    """

    def __init__(self, rel: "Relation", edge_idxs: np.ndarray):
        self.rel = rel
        self.edge_idxs = np.asarray(edge_idxs, dtype=int)

    def __len__(self):
        return int(self.edge_idxs.size)

    @property
    def size(self) -> int:
        return int(self.edge_idxs.size)

    def indices(self) -> np.ndarray:
        return self.edge_idxs

    def get_col(self, name: str) -> np.ndarray:
        return self.rel.get_attr(name)[self.edge_idxs]

    def get_attrs_view(self, names):
        return {n: self.get_col(n) for n in names}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get_col(key)

    def __setitem__(self, key: str, values):
        """按列名对视图选中边进行赋值。

        Args:
            key: 目标列名。
            values: 标量或与当前视图长度一致的一维数组。
        """
        self.set_attr(key, values)

    def set_attr(self, name: str, values, *, backend: str = 'auto'):
        """对当前视图选中的边执行列赋值。

        Args:
            name: 列名。
            values: 标量或长度与视图一致的一维数组。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        self.rel.set_attr(name, self.edge_idxs, values, backend=backend)

    def assign(self, values_by_attr: dict, *, backend: str = 'auto'):
        """对当前视图执行多列批量赋值。

        Args:
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        self.rel.assign(self.edge_idxs, values_by_attr, backend=backend)

    def __getattr__(self, item: str):
        if item in ("rel", "edge_idxs"):
            return super().__getattribute__(item)
        if hasattr(self.rel, "d") and item in self.rel.d:
            return self.get_col(item)
        raise AttributeError(item)


def dense_to_relation(mat: np.ndarray, src_pool, dst_pool=None, rel_name: str = 'relation', attr_name: str = 'weight', threshold: float = 0.0) -> Relation:
    """将稠密的 numpy 矩阵转换为 Relation（边表）。

    约定
    ----
    - 行对应 src_pool 的实体顺序（pos），列对应 dst_pool 的实体顺序（pos）。
    - 实际写入 Relation 时存储的是 uid（src_uid/dst_uid），以保证实体移动/压缩后关系仍稳定。
    - 仅把绝对值大于 threshold 的元作为边加入。

    返回
    ----
    - 新建的 Relation（拷贝）。
    """
    if dst_pool is None:
        dst_pool = src_pool
    if mat.ndim != 2:
        raise ValueError("mat 必须是二维数组")
    src_uids = np.asarray(src_pool.d['i'][:src_pool.size], dtype=np.int64)
    dst_uids = np.asarray(dst_pool.d['i'][:dst_pool.size], dtype=np.int64)
    if mat.shape[0] != src_uids.shape[0] or mat.shape[1] != dst_uids.shape[0]:
        raise ValueError("矩阵维度必须与源/目的实体数量匹配")

    rows, cols = np.nonzero(np.abs(mat) > threshold)
    if rows.size == 0:
        return Relation(rel_name, capacity=4)

    src_sel = src_uids[rows]
    dst_sel = dst_uids[cols]
    vals = mat[rows, cols]
    rel = Relation(rel_name, capacity=max(8, vals.size), attr_dtypes={attr_name: vals.dtype})
    rel.add(src_sel, dst_sel, **{attr_name: vals})
    return rel


class EntityPoolView:
    """EntityPool 的轻量视图：只保存 (pool, idxs)。

    - 不拷贝 records（避免一次性材料化多列）。
    - 需要哪列再取哪列：view['A']。

    注意：对列做 fancy indexing 通常会产生拷贝；view 的价值在于延迟取列与复用 idx。
    """

    def __init__(self, pool: "EntityPool", idxs: np.ndarray):
        self.pool = pool
        self.idxs = np.asarray(idxs, dtype=int)

    def __len__(self):
        return int(self.idxs.size)

    @property
    def size(self) -> int:
        return int(self.idxs.size)

    def indices(self) -> np.ndarray:
        return self.idxs

    def get_col(self, name: str) -> np.ndarray:
        return self.pool.get_attr(name)[self.idxs]

    def get_attrs_view(self, names):
        return {n: self.get_col(n) for n in names}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get_col(key)

    def __setitem__(self, key: str, values):
        """按列名对视图选中实体进行赋值。

        Args:
            key: 目标列名。
            values: 标量或与当前视图长度一致的一维数组。
        """
        self.set_attr(key, values)

    def set_attr(self, name: str, values, *, backend: str = 'auto'):
        """对当前视图选中的实体执行列赋值。

        Args:
            name: 列名。
            values: 标量或长度与视图一致的一维数组。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        self.pool.set_attr(name, self.idxs, values, backend=backend)

    def assign(self, values_by_attr: dict, *, backend: str = 'auto'):
        """对当前视图执行多列批量赋值。

        Args:
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        self.pool.assign(self.idxs, values_by_attr, backend=backend)


class EntityPool:
    """
    RECS 实体池（SoA 版），用于在不破坏向后兼容 API 的前提下，提供表格式实体存储、批量操作与关系扩展支撑。

    主要特性：
    - 使用字典 self.d 保存每列 ndarray，长度为 capacity
    - 自动维护 i (uid) 和 o (on) 两个必备字段
    - 支持 add / remove / enable / add_attribute / get_attr / get_attrs_view
    - 设计为可与旧有 ECSEngine API 兼容，以便逐步替换
    - 支持多后端计算（NumPy/PyTorch/MLX/JAX/TensorFlow）
    """

    def __init__(self, capacity, attr_dtypes, backend='numpy'):
        # 将 capacity 和 size 初始化
        self.capacity = int(capacity)
        self.size = 0  # 已分配的实体数量（逻辑行数）

        # 多后端支持：获取后端实例
        self.backend = _get_backend(backend)
        self._string_columns = set()  # 记录字符串列，强制使用 NumPy

        # d 存放每个属性对应的数组（storage dict）
        self.d = {}
        # _defaults 存放每个属性的默认值，用于扩容时初始化新区域
        self._defaults = {}

        # 强制添加 i(uid) 和 o(on) 两个必需字段
        attr_dtypes = dict(attr_dtypes)
        attr_dtypes['i'] = np.int64
        attr_dtypes['o'] = bool

        # i 的自增计数器
        self._i_counter = 0

        # 为每个属性创建底层数组
        for name, dtype in attr_dtypes.items():
            # 字符串列强制使用 NumPy
            if _is_string_dtype(dtype):
                arr = np.empty(self.capacity, dtype=dtype)
                self._string_columns.add(name)
            else:
                arr = self.backend.empty(self.capacity, dtype)
            # 保证为 C-contiguous（对 NumPy 后端）
            if self.backend.name == 'numpy':
                arr = np.ascontiguousarray(arr)
            self.d[name] = arr
            # 默认 default 为 None，用于在 add_attribute 或扩容时作为参考
            self._defaults[name] = None

        # 对 o 字段做默认初始化：默认全部 False（未激活）
        # JAX 数组不可变，需要使用 backend.zeros 创建新数组
        if self.backend.name == 'jax':
            self.d['o'] = self.backend.zeros(self.capacity, dtype=bool)
        else:
            self.d['o'][:] = False
        self._defaults['o'] = False
        # i 的默认值由 _i_counter 管理，defaults['i'] 设为 None
        self._defaults['i'] = None

        # 尝试为 i 字段批量填充连续 uid，确保底层数组有有效 uid
        try:
            uid_array = self.backend.arange(self._i_counter, self._i_counter + self.capacity, dtype=self.d['i'].dtype)
            if self.backend.name == 'jax':
                # JAX 数组不可变，直接替换整个数组
                self.d['i'] = uid_array
            else:
                self.d['i'][:self.capacity] = uid_array
            self._i_counter += self.capacity
        except Exception:
            # 若直接批量赋值失败，则逐个赋值（安全回退）
            for j in range(self.capacity):
                if self.backend.name == 'jax':
                    # JAX 需要使用 .at[].set() 语法
                    self.d['i'] = self.d['i'].at[j].set(int(self._i_counter))
                else:
                    self.d['i'][j] = int(self._i_counter)
                self._i_counter += 1

    def __len__(self):
        return int(self.size)

    # ------------------------------- 容量管理 -------------------------------
    def reserve(self, min_capacity):
        """
        确保 capacity 至少为 min_capacity，常用于提前分配以减少扩容次数。
        """
        if min_capacity > self.capacity:
            self._resize(int(min_capacity))

    def _resize(self, new_capacity):
        """
        扩容并重分配所有属性数组与 is_active（o）。
        """
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity:
            return
        old_capacity = self.capacity

        # 为每个属性重新分配底层数组并批量初始化
        for name, arr in list(self.d.items()):
            dtype = arr.dtype
            # 字符串列强制使用 NumPy
            is_string = name in self._string_columns or _is_string_dtype(dtype)
            if is_string:
                self._string_columns.add(name)
                new_arr = np.empty(new_capacity, dtype=dtype)
                # 如果原数组是 backend 数组，需要转换
                if not isinstance(arr, np.ndarray):
                    new_arr[:self.size] = self.backend.to_numpy(arr[:self.size])
                else:
                    new_arr[:self.size] = arr[:self.size]
            else:
                new_arr = self.backend.empty(new_capacity, dtype)
                new_arr[:self.size] = arr[:self.size]

            # 初始化新增尾部区间 [old_capacity:new_capacity)
            tail_len = new_capacity - old_capacity
            if tail_len > 0:
                if name == 'i':
                    # 为新增位置分配连续 uid
                    if is_string:
                        new_arr[old_capacity:new_capacity] = np.arange(self._i_counter, self._i_counter + tail_len, dtype=new_arr.dtype)
                    else:
                        new_arr[old_capacity:new_capacity] = self.backend.arange(self._i_counter, self._i_counter + tail_len, dtype=new_arr.dtype)
                    self._i_counter += tail_len
                elif name == 'o':
                    # o 字段默认 False（未激活）
                    new_arr[old_capacity:new_capacity] = False
                else:
                    # 其他字段按 _defaults 或类型推断进行初始化
                    default = self._defaults.get(name, None)
                    if default is not None:
                        # 尽量使用广播填充以提高效率
                        try:
                            new_arr[old_capacity:new_capacity] = default
                        except Exception:
                            # 若 dtype 为 object 或特殊类型，回退为逐元素赋值
                            new_arr[old_capacity:new_capacity] = default
                    else:
                        # 若没有默认值，则按 dtype 填充合理的空值
                        if is_string:
                            # 字符串列用 None
                            new_arr[old_capacity:new_capacity] = None
                        elif self.backend.is_floating(dtype) or self.backend.is_integer(dtype) or self.backend.is_bool(dtype):
                            # 数值/布尔类型用 0 / False
                            new_arr[old_capacity:new_capacity] = 0
                        else:
                            # 非数值类型用 None
                            new_arr[old_capacity:new_capacity] = None

            # 确保数组为 C-contiguous（仅 NumPy 后端）
            if self.backend.name == 'numpy':
                new_arr = np.ascontiguousarray(new_arr)
            self.d[name] = new_arr

        # 更新 capacity
        self.capacity = new_capacity

    # ------------------------------- 实体操作 -------------------------------
    def add(self, n=1, **kwargs):
        """
        批量添加 n 个实体，返回新增加的实体下标数组。
        """
        n = int(n)
        if n <= 0:
            return self.backend.empty(0, dtype=int)

        # 确保有足够空间
        if self.size + n > self.capacity:
            self._resize(max(self.capacity * 2, self.size + n))

        idxs = self.backend.arange(self.size, self.size + n, dtype=int)

        # 批量分配 uid
        uid_values = self.backend.arange(self._i_counter, self._i_counter + n, dtype=self.d['i'].dtype)
        if self.backend.name == 'jax':
            self.d['i'] = self.d['i'].at[idxs].set(uid_values)
        else:
            self.d['i'][idxs] = uid_values
        self._i_counter += n

        # 默认 o 为 True（批量添加后默认启用），但若 kwargs 中包含 o，则按其值设置
        if 'o' in kwargs:
            val_o = kwargs.pop('o')
            if np.isscalar(val_o):
                if self.backend.name == 'jax':
                    self.d['o'] = self.d['o'].at[idxs].set(val_o)
                else:
                    self.d['o'][idxs] = val_o
            else:
                arr_o = self.backend.asarray(val_o)
                if arr_o.shape[0] != n:
                    raise ValueError(f"Length mismatch for attribute 'o': expected {n}, got {arr_o.shape[0]}")
                if self.backend.name == 'jax':
                    self.d['o'] = self.d['o'].at[idxs].set(arr_o)
                else:
                    self.d['o'][idxs] = arr_o
        else:
            if self.backend.name == 'jax':
                self.d['o'] = self.d['o'].at[idxs].set(True)
            else:
                self.d['o'][idxs] = True

        # 处理其它属性（广播或按序列赋值）
        for name, value in kwargs.items():
            if name in ('i',):
                continue
            if name not in self.d:
                # 字符串值强制使用 NumPy 检测
                if _is_string_value(value):
                    sample = np.asarray(value)
                else:
                    sample = self.backend.asarray(value)
                dtype = sample.dtype
                self.add_attribute(name, dtype, default=0)
            arr = self.d[name]
            if np.isscalar(value):
                if name in self._string_columns:
                    # 字符串列是 NumPy，使用 NumPy 索引
                    idxs_np = self.backend.to_numpy(idxs)
                    arr[idxs_np] = value
                else:
                    self._set_item_with_immutable_check(name, idxs, value)
            else:
                if _is_string_value(value) or name in self._string_columns:
                    vals = np.asarray(value)
                    # 字符串列是 NumPy，使用 NumPy 索引
                    idxs_np = self.backend.to_numpy(idxs)
                    arr[idxs_np] = vals
                else:
                    # 使用目标数组的 dtype 转换值，避免类型不匹配警告
                    vals = self.backend.asarray(value, dtype=arr.dtype)
                    if vals.shape[0] != n:
                        raise ValueError(f"Length mismatch for attribute '{name}': expected {n}, got {vals.shape[0]}")
                    self._set_item_with_immutable_check(name, idxs, vals)

        # 注意：实体的存在由 size 管理；启用/激活由 d['o'] 管理
        self.size += n
        return idxs

    # ------------------------------- 属性管理 -------------------------------
    def add_attribute(self, name, dtype, default=0):
        """
        为当前 capacity 添加一个新字段（列），并用 default 初始化。
        """
        if name in self.d:
            raise ValueError(f"Attribute '{name}' already exists.")

        # 字符串列强制使用 NumPy
        if _is_string_dtype(dtype):
            self._string_columns.add(name)
            try:
                arr = np.full(self.capacity, default, dtype=dtype)
            except Exception:
                arr = np.empty(self.capacity, dtype=dtype)
                arr[:self.size] = default
                if self.size < self.capacity:
                    arr[self.size:] = default
            arr = np.ascontiguousarray(arr)
        else:
            try:
                arr = self.backend.full(self.capacity, default, dtype=dtype)
            except Exception:
                arr = self.backend.empty(self.capacity, dtype=dtype)
                arr[:self.size] = default
                if self.size < self.capacity:
                    arr[self.size:] = default

        self.d[name] = arr
        self._defaults[name] = default

    # ------------------------------- 辅助方法 -------------------------------
    @property
    def active_mask(self):
        """返回长度为 size 的布尔掩码，指示哪些实体处于启用/激活状态（基于 d['o']）。"""
        return self.d['o'][:self.size]

    def get_active_indices(self):
        """返回当前处于激活状态的实体索引。"""
        return self.backend.nonzero(self.active_mask)

    def is_active(self, idx):
        """检查指定索引的实体是否为激活状态。"""
        idxs = self._to_indices(idx)
        if len(idxs) == 0:
            return False
        return bool(self.backend.all(self.d['o'][idxs]))

    # ------------------------------- 数据导出 -------------------------------

    def to_numpy(self):
        """将所有列转换为 NumPy 数组并返回字典。

        便捷方法：用于将数据从其他后端（PyTorch/MLX/JAX/TF）导出为 NumPy 格式。
        """
        result = {}
        for name, arr in self.d.items():
            if name in self._string_columns:
                result[name] = arr[:self.size]
            else:
                result[name] = self.backend.to_numpy(arr[:self.size])
        return result

    # ------------------------------- 索引与查询 -------------------------------
    def get_attr(self, name):
        """获取指定属性的数组视图（仅前 size 个元素）。"""
        if name not in self.d:
            raise KeyError(f"Attribute '{name}' not found")
        return self.d[name][:self.size]

    def get_attrs_view(self, names):
        """获取多个属性的视图字典。"""
        return {name: self.get_attr(name) for name in names}

    def _to_indices(self, idx):
        """将多种索引输入统一为整型索引数组。
        支持：int、slice、布尔掩码、整型数组/可迭代。
        """
        # 单个整数
        if isinstance(idx, (int, np.integer)):
            return self.backend.array([int(idx)], dtype=int)

        # slice
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            return self.backend.arange(start, stop, step, dtype=int)

        arr = self.backend.asarray(idx)
        # 布尔掩码
        if self.backend.is_bool(arr.dtype):
            if arr.shape[0] != self.size:
                raise ValueError(f"Boolean mask length must equal current size ({self.size})")
            return self.backend.nonzero(arr)

        # 整型数组或可迭代
        if self.backend.is_integer(arr.dtype):
            return self.backend.astype(arr, int)

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def remove(self, idx):
        """remove = disable：禁用指定索引的实体（标记为 inactive）。

        注意：为了保持数组连续性，这里只标记为 inactive，不真正删除。
        """
        self.disable(idx)

    def disable(self, idx):
        """禁用指定索引的实体（将 o 字段设为 False）。"""
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return
        self._set_item_with_immutable_check('o', idxs, False)

    def enable(self, idx):
        """启用指定索引的实体（将 o 字段设为 True）。"""
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return
        self._set_item_with_immutable_check('o', idxs, True)

    def set_attr(self, name, idx, values, *, backend='auto'):
        """按索引对单列执行赋值（类 NumPy 语义）。

        Args:
            name: 列名。
            idx: 目标索引，支持 int/slice/布尔掩码/整型数组。
            values: 标量或与索引长度一致的一维数组。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。

        Raises:
            KeyError: 列不存在。
            ValueError: 向量赋值长度与索引长度不一致。
        """
        _normalize_backend(backend)
        if name not in self.d:
            raise KeyError(f"Attribute '{name}' not found")
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return

        # 处理标量值
        if np.isscalar(values):
            self._set_item_with_immutable_check(name, idxs, values)
            return

        # 处理数组值
        if name in self._string_columns:
            vals = np.asarray(values)
            # 字符串列是 NumPy，使用 NumPy 索引
            idxs_np = self.backend.to_numpy(idxs)
            self.d[name][idxs_np] = vals
        else:
            vals = self.backend.asarray(values)
            # 使用 shape[0] 而不是 size，因为 PyTorch tensor 的 size 是方法
            idxs_len = idxs.shape[0] if hasattr(idxs, 'shape') else len(idxs)
            if vals.shape[0] != idxs_len:
                raise ValueError(f"Length mismatch: expected {idxs_len}, got {vals.shape[0]}")
            self._set_item_with_immutable_check(name, idxs, vals)

    def _set_item_with_immutable_check(self, name, idxs, values):
        """处理不可变数组的赋值（如 JAX）。"""
        arr = self.d[name]
        try:
            arr[idxs] = values
        except TypeError as e:
            # 可能是不可变数组（如 JAX）
            if 'immutable' in str(e).lower() or 'not support item assignment' in str(e).lower():
                # 使用 .at[idx].set(y) 模式
                if hasattr(arr, 'at'):
                    if np.isscalar(values):
                        self.d[name] = arr.at[idxs].set(values)
                    else:
                        self.d[name] = arr.at[idxs].set(values)
                else:
                    raise
            else:
                raise

    def assign(self, idx, values_by_attr, *, backend='auto'):
        """对指定索引的实体执行多列批量赋值。

        Args:
            idx: 目标索引，支持 int/slice/布尔掩码/整型数组。
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
        """
        for name, values in values_by_attr.items():
            self.set_attr(name, idx, values, backend=backend)

    def index(self, pred=None, *, return_='indices', include_active_only=False):
        """索引层：根据谓词返回实体索引或掩码。

        Args:
            pred: 谓词函数或布尔掩码。若为 None，返回全部。
            return_: 'indices' 返回整数索引数组，'mask' 返回布尔掩码。
            include_active_only: 是否只考虑激活状态的实体。

        Returns:
            np.ndarray: 整数索引数组或布尔掩码。
        """
        if return_ not in ('indices', 'mask'):
            raise ValueError("return_ must be 'indices' or 'mask'")

        # 基础掩码
        if include_active_only:
            # 转换为 numpy 并复制，兼容多后端
            mask = self.backend.to_numpy(self.active_mask).copy()
        else:
            mask = np.ones(self.size, dtype=bool)

        # 应用谓词
        if pred is not None:
            if callable(pred):
                # 谓词函数接收 EntityPool 自身作为参数
                pred_mask = pred(self)
            else:
                pred_mask = pred
            # 转换为 NumPy 数组
            pred_mask = self.backend.to_numpy(pred_mask).astype(bool)
            if pred_mask.shape[0] != self.size:
                raise ValueError(f"Predicate mask length must equal size ({self.size})")
            mask &= pred_mask

        if return_ == 'mask':
            return mask
        return np.nonzero(mask)[0]

    def query(self, pred=None, *, return_='indices', include_active_only=False):
        """查询层：根据谓词返回实体索引或视图。

        Args:
            pred: 谓词函数或布尔掩码。若为 None，返回全部。
            return_: 'indices' 返回整数索引数组，'view' 返回 EntityPoolView。
            include_active_only: 是否只考虑激活状态的实体。

        Returns:
            np.ndarray 或 EntityPoolView。
        """
        idxs = self.index(pred, return_='indices', include_active_only=include_active_only)

        if return_ == 'indices':
            return idxs
        if return_ == 'view':
            return EntityPoolView(self, idxs)
        if return_ == 'records':
            # 物化拷贝：返回字典，每个列按 idxs 索引并转为 numpy
            result = {}
            for k in self.d:
                arr = self.d[k]
                if k in self._string_columns:
                    result[k] = arr[idxs].copy()
                else:
                    result[k] = self.backend.to_numpy(arr[idxs])
            return result

        raise ValueError("return_ must be 'indices', 'view', or 'records'")

    def where_set(self, pred, values_by_attr, *, backend='auto', include_active_only=False):
        """条件批量赋值：对满足谓词的实体设置属性值。

        Args:
            pred: 谓词函数或布尔掩码。
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
            include_active_only: 是否只考虑激活状态的实体。

        Returns:
            np.ndarray: 满足条件的实体索引数组。
        """
        idxs = self.index(pred, return_='indices', include_active_only=include_active_only)
        if idxs.size == 0:
            return idxs
        self.assign(idxs, values_by_attr, backend=backend)
        return idxs


class RECS:
    """
    RECS 向后兼容的 ECSEngine 包装器，内部委托给 EntityPool 来管理存储。

    该类保持原有 API surface（例如访问 d）以便旧的 demo/脚本继续工作。
    支持多后端计算（NumPy/PyTorch/MLX/JAX/TensorFlow）。
    """

    def __init__(self, capacity, attr_dtypes, backend='numpy'):
        # 创建内部 EntityPool 并将其存储暴露为 self.d 以兼容旧代码
        self._pool = EntityPool(capacity=capacity, attr_dtypes=attr_dtypes, backend=backend)
        # 直接暴露 d 以便旧示例仍然可以操作 et.d[...] 保持工作
        self.d = self._pool.d
        # 暴露 backend 引用
        self.backend = self._pool.backend

    # proxy commonly used attributes and methods to the internal pool
    @property
    def capacity(self):
        return self._pool.capacity

    @capacity.setter
    def capacity(self, v):
        self._pool.capacity = v

    @property
    def size(self):
        return self._pool.size

    def reserve(self, min_capacity):
        return self._pool.reserve(min_capacity)

    def add(self, n=1, **kwargs):
        return self._pool.add(n, **kwargs)

    def remove(self, idx):
        return self._pool.remove(idx)

    def enable(self, idx):
        return self._pool.enable(idx)

    def disable(self, idx):
        self._pool.disable(idx)

    def add_attribute(self, name, dtype, default=0):
        return self._pool.add_attribute(name, dtype, default)

    def get_attr(self, name):
        return self._pool.get_attr(name)

    def get_attrs_view(self, names):
        return self._pool.get_attrs_view(names)

    @property
    def active_mask(self):
        return self._pool.active_mask

    def get_active_indices(self):
        return self._pool.get_active_indices()

    def is_active(self, idx):
        """检查指定索引的实体是否为激活状态。"""
        return self._pool.is_active(idx)

    def uid_array(self):
        """返回当前已分配实体的 uid 数组视图。"""
        return self._pool.d['i'][:self._pool.size]

    def dense_matrix_to_relation(self, mat, dst_pool=None, relation_name='relation', attr_name='weight', threshold=0.0):
        """
        便捷包装：将以本引擎实体池为行的稠密矩阵转换为 RECS Relation。
        """
        if dst_pool is None:
            dst_pool = self
        dst_for_rel = dst_pool._pool if isinstance(dst_pool, RECS) else dst_pool
        return dense_to_relation(mat, src_pool=self._pool, dst_pool=dst_for_rel, rel_name=relation_name, attr_name=attr_name, threshold=threshold)

    def __len__(self):
        return int(self._pool.size)

    def index(self, pred=None, *, return_='indices', include_active_only=False):
        return self._pool.index(pred, return_=return_, include_active_only=include_active_only)

    def query(self, pred=None, *, return_='indices', include_active_only=False):
        return self._pool.query(pred, return_=return_, include_active_only=include_active_only)

    def set_attr(self, name, idx, values, *, backend='auto'):
        return self._pool.set_attr(name, idx, values, backend=backend)

    def assign(self, idx, values_by_attr, *, backend='auto'):
        return self._pool.assign(idx, values_by_attr, backend=backend)

    def where_set(self, pred, values_by_attr, *, backend='auto', include_active_only=False):
        return self._pool.where_set(pred, values_by_attr, backend=backend, include_active_only=include_active_only)

    def to_numpy(self):
        return self._pool.to_numpy()


# 模块导出（单文件实现）
__all__ = [
    'to_indices',
    'to_mask',
    'SparseBoolBitset',
    'SparseNullable',
    'DenseRelation',
    'SparseRelationEdges',
    'Relation',
    'RelationView',
    'dense_to_relation',
    'EntityPool',
    'EntityPoolView',
    'RECS',
]
