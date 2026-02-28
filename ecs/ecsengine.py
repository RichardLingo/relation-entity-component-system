import numpy as np
from typing import Optional

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
# 第二类二维数组：Relation（稠密/稀疏后端）
# =============================


class DenseRelation:
    """稠密关系：用二维 ndarray 表示。"""

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
    """稀疏关系：边表（src_uid/dst_uid/value）。

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
    边表关系（SoA 风格）：以 src_uid / dst_uid 为主键存储，其他属性按列保存。

    设计原则：
    - 存储 uid 而非 pool 引用，以便实体可以移动/压缩而关系保持稳定。
    - 提供批量添加、按 src/dst 查询，以及能从稠密矩阵构造等功能。
    """

    def __init__(self, name: str, capacity: int = 1024, attr_dtypes: Optional[dict] = None):
        self.name = name
        self.capacity = int(capacity)
        self.size = 0
        self._capacity = self.capacity
        self.d = {}
        self.d['src_uid'] = np.empty(self.capacity, dtype=np.int64)
        self.d['dst_uid'] = np.empty(self.capacity, dtype=np.int64)
        if attr_dtypes:
            for k, dt in attr_dtypes.items():
                if k in ('src_uid', 'dst_uid'):
                    continue
                self.d[k] = np.empty(self.capacity, dtype=dt)

    def _resize(self, new_cap: int):
        if new_cap <= self._capacity:
            return
        old = self._capacity
        for k, arr in list(self.d.items()):
            new_arr = np.empty(new_cap, dtype=arr.dtype)
            new_arr[:self.size] = arr[:self.size]
            # 尾部初始化
            if np.issubdtype(new_arr.dtype, np.floating) or np.issubdtype(new_arr.dtype, np.integer) or np.issubdtype(new_arr.dtype, np.bool_):
                new_arr[old:new_cap] = 0
            else:
                new_arr[old:new_cap] = None
            self.d[k] = np.ascontiguousarray(new_arr)
        self._capacity = new_cap

    def add(self, src_uids, dst_uids, **attrs):
        src_u = np.asarray(src_uids, dtype=np.int64)
        dst_u = np.asarray(dst_uids, dtype=np.int64)
        if src_u.shape != dst_u.shape:
            raise ValueError("src and dst must have same shape")
        n = src_u.size
        if n == 0:
            return np.empty(0, dtype=int)
        if self.size + n > self._capacity:
            self._resize(max(self._capacity * 2, self.size + n))
        idxs = np.arange(self.size, self.size + n, dtype=int)
        self.d['src_uid'][idxs] = src_u
        self.d['dst_uid'][idxs] = dst_u
        for k, v in attrs.items():
            if k not in self.d:
                arr = np.empty(self._capacity, dtype=np.asarray(v).dtype)
                # 初始化尾部
                if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                    arr[self.size:self._capacity] = 0
                else:
                    arr[self.size:self._capacity] = None
                self.d[k] = arr
            val = np.asarray(v)
            if val.size == 1:
                self.d[k][idxs] = val.item()
            else:
                if val.shape[0] != n:
                    raise ValueError(f"Length mismatch for edge attr {k}")
                self.d[k][idxs] = val
        self.size += n
        return idxs

    def remove_by_indices(self, idxs):
        idxs = np.unique(np.asarray(idxs, dtype=int))
        valid = (idxs >= 0) & (idxs < self.size)
        if not np.all(valid):
            raise IndexError("Index out of bounds")
        mask = np.ones(self.size, dtype=bool)
        mask[idxs] = False
        for k in self.d:
            self.d[k][:mask.sum()] = self.d[k][mask]
        self.size = mask.sum()

    def neighbors_from_uid(self, src_uid):
        mask = self.d['src_uid'][:self.size] == src_uid
        dsts = self.d['dst_uid'][:self.size][mask]
        attrs = {k: v[:self.size][mask] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        return dsts, attrs

    def neighbors_to_uid(self, dst_uid):
        mask = self.d['dst_uid'][:self.size] == dst_uid
        srcs = self.d['src_uid'][:self.size][mask]
        attrs = {k: v[:self.size][mask] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        return srcs, attrs

    def add_attribute(self, name, dtype, default=0):
        if name in self.d:
            raise ValueError(f"Attribute '{name}' already exists.")
        try:
            arr = np.full(self._capacity, default, dtype=dtype)
        except Exception:
            arr = np.empty(self._capacity, dtype=dtype)
            arr[:self.size] = default
            if self.size < self._capacity:
                arr[self.size:self._capacity] = default
        self.d[name] = np.ascontiguousarray(arr)

    def get_attr(self, name):
        return self.d[name][:self.size]

    def get_attrs_view(self, names):
        return {name: self.get_attr(name) for name in names}

    def _to_indices(self, idx):
        if isinstance(idx, (int, np.integer)):
            return np.array([int(idx)], dtype=int)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            return np.arange(start, stop, step, dtype=int)
        arr = np.asarray(idx)
        if arr.dtype == np.bool_:
            if arr.shape[0] != self.size:
                raise ValueError(f"Boolean mask length must equal current size ({self.size})")
            return np.nonzero(arr)[0]
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(int)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def indices_from_uid(self, src_uid):
        return np.nonzero(self.d['src_uid'][:self.size] == src_uid)[0]

    def indices_to_uid(self, dst_uid):
        return np.nonzero(self.d['dst_uid'][:self.size] == dst_uid)[0]

    def take(self, idx):
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return Relation(self.name, capacity=4)
        attr_dtypes = {k: v.dtype for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        rel = Relation(self.name, capacity=max(8, idxs.size), attr_dtypes=attr_dtypes)
        src_u = self.d['src_uid'][idxs]
        dst_u = self.d['dst_uid'][idxs]
        attrs = {k: v[idxs] for k, v in self.d.items() if k not in ('src_uid', 'dst_uid')}
        rel.add(src_u, dst_u, **attrs)
        return rel

    def remove_by_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.size:
            raise ValueError(f"Mask length must equal current size ({self.size})")
        keep = ~mask
        for k in self.d:
            self.d[k][:keep.sum()] = self.d[k][keep]
        self.size = int(keep.sum())

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
        val = np.asarray(values)
        if val.size == 1:
            self.d[name][idxs] = val.item()
        else:
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
        order = np.argsort(arr, kind='mergesort')
        return order if ascending else order[::-1]

    def sort_by(self, name, ascending: bool = True):
        order = self.argsort_by(name, ascending=ascending)
        for k in self.d:
            self.d[k][:self.size] = self.d[k][:self.size][order]
        return order

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
    def _map_uids_to_pos(uids: np.ndarray, uid_to_pos):
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
    SoA 实体池（轻量版），用于在不破坏向后兼容 API 的前提下，提供表格式实体存储与批量操作。

    主要特性：
    - 使用字典 self.d 保存每列 ndarray，长度为 capacity
    - 自动维护 i (uid) 和 o (on) 两个必备字段
    - 支持 add / remove / enable / add_attribute / get_attr / get_attrs_view
    - 设计为可与旧有 ECSEngine API 兼容，以便逐步替换
    """

    def __init__(self, capacity, attr_dtypes):
        # 将 capacity 和 size 初始化
        self.capacity = int(capacity)
        self.size = 0  # 已分配的实体数量（逻辑行数）

        # d 存放每个属性对应的 numpy 数组（storage dict）
        self.d = {}
        # _defaults 存放每个属性的默认值，用于扩容时初始化新区域
        self._defaults = {}

        # 强制添加 i(uid) 和 o(on) 两个必需字段
        attr_dtypes = dict(attr_dtypes)
        attr_dtypes['i'] = np.int64
        attr_dtypes['o'] = bool

        # i 的自增计数器
        self._i_counter = 0

        # 为每个属性创建底层数组，使用 np.empty 并保证为 C-contiguous
        for name, dtype in attr_dtypes.items():
            arr = np.empty(self.capacity, dtype=dtype)
            # 先不对数组内容进行逐元素赋值，仅保证内存已分配
            self.d[name] = np.ascontiguousarray(arr)
            # 默认 default 为 None，用于在 add_attribute 或扩容时作为参考
            self._defaults[name] = None

        # 对 o 字段做默认初始化：默认全部 False（未激活）
        self.d['o'][:] = False
        self._defaults['o'] = False
        # i 的默认值由 _i_counter 管理，defaults['i'] 设为 None
        self._defaults['i'] = None

        # 尝试为 i 字段批量填充连续 uid，确保底层数组有有效 uid
        try:
            self.d['i'][:self.capacity] = np.arange(self._i_counter, self._i_counter + self.capacity, dtype=self.d['i'].dtype)
            self._i_counter += self.capacity
        except Exception:
            # 若直接批量赋值失败，则逐个赋值（安全回退）
            for j in range(self.capacity):
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
            # 分配新的底层数组并拷贝已有数据
            new_arr = np.empty(new_capacity, dtype=dtype)
            new_arr[:self.size] = arr[:self.size]

            # 初始化新增尾部区间 [old_capacity:new_capacity)
            tail_len = new_capacity - old_capacity
            if tail_len > 0:
                if name == 'i':
                    # 为新增位置分配连续 uid
                    new_arr[old_capacity:new_capacity] = np.arange(self._i_counter, self._i_counter + tail_len, dtype=new_arr.dtype)
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
                        if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                            # 数值/布尔类型用 0 / False
                            new_arr[old_capacity:new_capacity] = 0
                        else:
                            # 非数值类型用 None
                            new_arr[old_capacity:new_capacity] = None

            # 确保数组为 C-contiguous
            self.d[name] = np.ascontiguousarray(new_arr)

        # 更新 capacity
        self.capacity = new_capacity

    # ------------------------------- 实体操作 -------------------------------
    def add(self, n=1, **kwargs):
        """
        批量添加 n 个实体，返回新增加的实体下标数组。
        """
        n = int(n)
        if n <= 0:
            return np.empty(0, dtype=int)

        # 确保有足够空间
        if self.size + n > self.capacity:
            self._resize(max(self.capacity * 2, self.size + n))

        idxs = np.arange(self.size, self.size + n, dtype=int)

        # 批量分配 uid
        self.d['i'][idxs] = np.arange(self._i_counter, self._i_counter + n, dtype=self.d['i'].dtype)
        self._i_counter += n

        # 默认 o 为 True（批量添加后默认启用），但若 kwargs 中包含 o，则按其值设置
        if 'o' in kwargs:
            val_o = kwargs.pop('o')
            if np.isscalar(val_o):
                self.d['o'][idxs] = val_o
            else:
                arr_o = np.asarray(val_o)
                if arr_o.shape[0] != n:
                    raise ValueError(f"Length mismatch for attribute 'o': expected {n}, got {arr_o.shape[0]}")
                self.d['o'][idxs] = arr_o
        else:
            self.d['o'][idxs] = True

        # 处理其它属性（广播或按序列赋值）
        for name, value in kwargs.items():
            if name in ('i',):
                continue
            if name not in self.d:
                sample = np.asarray(value)
                dtype = sample.dtype
                self.add_attribute(name, dtype, default=0)
            arr = self.d[name]
            if np.isscalar(value):
                arr[idxs] = value
            else:
                vals = np.asarray(value)
                if vals.shape[0] != n:
                    raise ValueError(f"Length mismatch for attribute '{name}': expected {n}, got {vals.shape[0]}")
                arr[idxs] = vals

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

        try:
            arr = np.full(self.capacity, default, dtype=dtype)
        except Exception:
            # 若直接创建失败，则先创建空数组再填充值
            arr = np.empty(self.capacity, dtype=dtype)
            arr[:self.size] = default
            if self.size < self.capacity:
                arr[self.size:] = default

        arr = np.ascontiguousarray(arr)
        self.d[name] = arr
        self._defaults[name] = default

    # ------------------------------- 辅助方法 -------------------------------
    @property
    def active_mask(self):
        """返回长度为 size 的布尔掩码，指示哪些实体处于启用/激活状态（基于 d['o']）。"""
        return self.d['o'][:self.size]

    def get_active_indices(self):
        """返回当前激活的实体的下标数组（与 active_mask 一致）。"""
        return np.nonzero(self.active_mask)[0]

    def get_attr(self, name):
        """返回属性 name 的视图（只包含前 size 项）。注意：对布尔掩码的索引会返回拷贝。"""
        return self.d[name][:self.size]

    def get_attrs_view(self, names):
        """
        批量获取多个属性的视图，返回字典（每个值均为长度为 size 的 ndarray 视图）。
        便于对多个列进行向量化计算。
        """
        return {name: self.get_attr(name) for name in names}

    def _to_indices(self, idx):
        """
        将多种索引输入统一为整型索引数组（np.ndarray of int）。
        支持：int、slice、布尔掩码、整型数组/可迭代。
        """
        # 单个整数
        if isinstance(idx, (int, np.integer)):
            return np.array([int(idx)], dtype=int)

        # slice
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            return np.arange(start, stop, step, dtype=int)

        arr = np.asarray(idx)
        # 布尔掩码
        if arr.dtype == np.bool_:
            if arr.shape[0] != self.size:
                raise ValueError(f"Boolean mask length must equal current size ({self.size})")
            return np.nonzero(arr)[0]

        # 整型数组或可迭代
        if np.issubdtype(arr.dtype, np.integer):
            res = arr.astype(int)
            return res

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def remove(self, idx):
        """
        将指定的下标集合标记为 inactive，并将 o 设为 False。
        """
        idxs = self._to_indices(idx)
        # 边界检查
        if idxs.size == 0:
            return
        if np.any((idxs < 0) | (idxs >= self.size)):
            raise IndexError("Index out of bounds")
        # 向量化设置：只操作 o（激活标志）
        self.d['o'][idxs] = False

    def enable(self, idx):
        """
        将指定的下标集合标记为 active，并将 o 设为 True。
        """
        idxs = self._to_indices(idx)
        if idxs.size == 0:
            return
        if np.any((idxs < 0) | (idxs >= self.size)):
            raise IndexError("Index out of bounds")
        # 启用实体：设置 o 为 True
        self.d['o'][idxs] = True

    def query(self, pred=None, *, return_: str = 'indices', include_active_only: bool = False):
        """高阶查询：根据实体属性谓词返回 indices/mask/records/view。

        return_:
            - 'indices'：返回整型索引数组（轻量，不拷贝列数据）
            - 'mask'：返回布尔掩码（轻量，不拷贝列数据）
            - 'records'：返回 dict[str, np.ndarray]（拷贝）
            - 'view'：返回 EntityPoolView（只保存 idx + pool 引用）
        """
        size = self.size
        if size == 0:
            if return_ == 'mask':
                return np.zeros(0, dtype=bool)
            if return_ == 'indices':
                return np.empty(0, dtype=int)
            if return_ == 'records':
                return {k: v[:0] for k, v in self.d.items()}
            if return_ == 'view':
                return EntityPoolView(self, np.empty(0, dtype=int))
            raise ValueError("return_ 必须为 'indices'/'mask'/'records'/'view'")

        if pred is None:
            mask = np.ones(size, dtype=bool)
        else:
            mask = pred(self) if callable(pred) else pred
            mask = np.asarray(mask, dtype=bool)
            if mask.shape[0] != size:
                raise ValueError(f"pred 掩码长度必须等于 pool.size ({size})")

        if include_active_only:
            mask = mask & self.active_mask

        if return_ == 'mask':
            return mask

        idxs = np.nonzero(mask)[0]
        if return_ == 'indices':
            return idxs

        if return_ == 'view':
            return EntityPoolView(self, idxs)

        if return_ == 'records':
            # 注意：fancy indexing 返回拷贝
            return {k: v[:size][idxs] for k, v in self.d.items()}

        raise ValueError("return_ 必须为 'indices'/'mask'/'records'/'view'")

    def index(self, pred=None, *, return_: str = 'indices', include_active_only: bool = False):
        """索引层：只返回实体位置（indices/mask）。

        这是 `EntityPool.query(...)` 的轻量子集，不返回 records/view。
        """
        if return_ not in ('indices', 'mask'):
            raise ValueError("EntityPool.index 的 return_ 只支持 'indices' 或 'mask'")
        return self.query(pred, return_=return_, include_active_only=include_active_only)

    def set_attr(self, name, idx, values, *, backend: str = 'auto'):
        """按索引对单列执行赋值（类 NumPy 语义）。

        Args:
            name: 列名。
            idx: 目标行索引，支持 int/slice/布尔掩码/整型数组。
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
        val = np.asarray(values)
        if val.size == 1:
            self.d[name][idxs] = val.item()
            return
        if val.shape[0] != idxs.size:
            raise ValueError(f"Length mismatch for attribute '{name}': expected {idxs.size}, got {val.shape[0]}")
        self.d[name][idxs] = val

    def assign(self, idx, values_by_attr: dict, *, backend: str = 'auto'):
        """按同一索引对多列执行批量赋值。

        Args:
            idx: 目标行索引，支持 int/slice/布尔掩码/整型数组。
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
            pred,
            values_by_attr: dict,
            *,
            backend: str = 'auto',
            include_active_only: bool = False
    ):
        """按查询条件对多列执行批量赋值。

        Args:
            pred: 查询条件，可为布尔掩码或 ``callable(pool) -> mask``。
            values_by_attr: 列名到赋值内容的映射。
            backend: 后端类型，支持 ``'auto'``、``'dense'``、``'sparse'``。
            include_active_only: 是否仅对激活实体生效。
        """
        _normalize_backend(backend)
        idxs = self.query(pred, return_='indices', include_active_only=include_active_only)
        self.assign(idxs, values_by_attr, backend=backend)
        return idxs


class ECSEngine:
    """
    向后兼容的 ECSEngine 包装器，内部委托给 EntityPool 来管理存储。

    该类保持原有 API surface（例如访问 d）以便旧的 demo/脚本继续工作。
    """

    def __init__(self, capacity, attr_dtypes):
        # 创建内部 EntityPool 并将其存储暴露为 self.d 以兼容旧代码
        self._pool = EntityPool(capacity=capacity, attr_dtypes=attr_dtypes)
        # 直接暴露 d 以便旧示例仍然可以操作 et.d[...] 保持工作
        self.d = self._pool.d

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

    def uid_array(self):
        """返回当前已分配实体的 uid 数组视图。"""
        return self._pool.d['i'][:self._pool.size]

    def dense_matrix_to_relation(self, mat: np.ndarray, dst_pool=None, relation_name: str = 'relation', attr_name: str = 'weight', threshold: float = 0.0) -> Relation:
        """
        便捷包装：将以本引擎的实体池为行的稠密矩阵转换为 Relation。
        如果 dst_pool 为 None，则使用 self 作为目的池。
        """
        if dst_pool is None:
            dst_pool = self
        # if dst_pool is an ECSEngine, pass its internal pool; otherwise assume it's a pool-like object
        dst_for_rel = dst_pool._pool if isinstance(dst_pool, ECSEngine) else dst_pool
        return dense_to_relation(mat, src_pool=self._pool, dst_pool=dst_for_rel, rel_name=relation_name, attr_name=attr_name, threshold=threshold)

    def __len__(self):
        return int(self._pool.size)

    def index(self, pred=None, *, return_: str = 'indices', include_active_only: bool = False):
        """索引层：只返回实体位置（indices/mask）。"""
        return self._pool.index(pred, return_=return_, include_active_only=include_active_only)

    def query(self, pred=None, *, return_: str = 'indices', include_active_only: bool = False):
        """查询层：返回 indices/mask/view/records。"""
        return self._pool.query(pred, return_=return_, include_active_only=include_active_only)

    def set_attr(self, name, idx, values, *, backend: str = 'auto'):
        """代理：按索引对单列执行赋值。"""
        return self._pool.set_attr(name, idx, values, backend=backend)

    def assign(self, idx, values_by_attr: dict, *, backend: str = 'auto'):
        """代理：按索引对多列执行批量赋值。"""
        return self._pool.assign(idx, values_by_attr, backend=backend)

    def where_set(self, pred, values_by_attr: dict, *, backend: str = 'auto', include_active_only: bool = False):
        """代理：按查询条件对多列执行批量赋值。"""
        return self._pool.where_set(pred, values_by_attr, backend=backend, include_active_only=include_active_only)


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
    'ECSEngine',
]
