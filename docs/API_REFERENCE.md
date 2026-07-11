# API 参考（面向用户）

本文件为面向用户的 API 参考，重点说明常用类、方法与典型用法。它使用简洁的语义化描述并包含小示例，帮助用户快速查阅。

目录

- RECS（高层）
- EntityPool / EntityPoolView
- Relation / RelationView / SparseRelationEdges / DenseRelation
- 工具函数（to_indices、to_mask、dense_to_relation）


---

RECS（高层）

构造

```
RECS(capacity, attr_dtypes, backend='numpy')
```

- capacity: 初始容量（整数）。
- attr_dtypes: 字典，键为属性名，值为 numpy dtype（字符串或 dtype 对象）。
- backend: 计算后端，支持 'numpy'（默认）/ 'torch' / 'mlx' / 'jax' / 'auto'。

说明：RECS 是对 `EntityPool` 的包装，提供实体表与关系扩展的一体化入口。

常用方法

- add(n=1, **kwargs) -> np.ndarray[int]
  - 批量添加 n 个实体，返回新加入的行下标数组（在 pool 中的索引）。
  - kwargs 可指定每个新增实体的属性（支持标量广播或与 n 长度一致的数组）。

- remove(idx)
  - 逻辑禁用指定下标（或下标数组 / 布尔掩码）。仅设置 `o` 字段为 False。

- enable(idx)
  - 将指定下标标记为启用（`o` = True）。

- disable(idx)
  - 将指定下标标记为禁用（`o` = False）。

- is_active(idx) -> bool
  - 检查指定索引的实体是否为激活状态。

- add_attribute(name, dtype, default=0)
  - 动态添加列并用 default 初始化。

- get_attr(name) -> np.ndarray | backend array
  - 返回属性 name 的视图（长度为当前 size），数据类型取决于后端。

- get_attrs_view(names) -> dict
  - 返回多个属性的视图。

- index(pred=None, *, return_='indices', include_active_only=False)
  - 轻量索引层：只返回位置（indices/mask）。
  - pred: `callable(pool) -> mask` 或布尔掩码；当为 None 时表示全选。

- query(pred=None, *, return_='indices', include_active_only=False)
  - 查询层：可返回 'indices' / 'mask' / 'view' / 'records'。
  - 'view' 返回 `EntityPoolView`（延迟取列）；'records' 返回拷贝的字典集合。

- set_attr(name, idx, values, *, backend='auto')
  - 按索引对单列赋值，支持标量/一维数组赋值（行为与 NumPy 类似）。

- assign(idx, values_by_attr: dict, *, backend='auto')
  - 在同一索引上对多列批量赋值（values_by_attr: 列名->值）。

- where_set(pred, values_by_attr: dict, *, backend='auto', include_active_only=False)
  - 按谓词直接对命中实体批量赋值（等价于先 query(..., 'indices') 再 assign）。

- to_numpy() -> dict[str, np.ndarray]
  - 将所有列导出为 NumPy 数组并返回字典。无论使用哪个后端，均可通过此方法统一导出。

- uid_array() -> np.ndarray
  - 返回当前已分配实体（前 size 项）的 uid 数组视图。

- dense_matrix_to_relation(mat, dst_pool=None, relation_name='relation', attr_name='weight', threshold=0.0) -> Relation
  - 将稠密矩阵按行/列映射为边表（只添加绝对值 > threshold 的元素），返回新 Relation。


EntityPool / EntityPoolView

EntityPool

构造

```
EntityPool(capacity, attr_dtypes, backend='numpy')
```

- capacity: 初始容量（整数）。
- attr_dtypes: 字典，键为属性名，值为 dtype（支持 numpy dtype 或 'U10' 等字符串类型）。
- backend: 计算后端，支持 'numpy'（默认）/ 'torch' / 'mlx' / 'jax' / 'auto'。

常用属性/方法（用户层）

- d: dict[str, np.ndarray | backend array]
  - 底层列存储；`d['i']` 是 uid 列，`d['o']` 是激活位（bool）。数据类型取决于后端。

- capacity, size

- add(n=1, **kwargs) -> np.ndarray[int]
  - 参见 RECS.add（EntityPool 提供实际实现）。

- remove(idx) / enable(idx)

- is_active(idx) -> bool
  - 检查指定索引的实体是否为激活状态。

- add_attribute(name, dtype, default=0)

- get_attr(name) -> np.ndarray | backend array
  - 返回长度为 size 的列视图。

- get_attrs_view(names) -> dict

- index(pred=None, *, return_='indices', include_active_only=False)
  - 只返回位置（indices/mask），语义与 RECS.index 相同。

- query(pred=None, *, return_='indices', include_active_only=False)
  - 返回 'indices' / 'mask' / 'view' / 'records'。

- set_attr(name, idx, values, *, backend='auto')
  - 按索引修改列值（支持标量/一维数组）。

- assign(idx, values_by_attr: dict, *, backend='auto')

- where_set(pred, values_by_attr: dict, *, backend='auto', include_active_only=False)

- to_numpy() -> dict[str, np.ndarray]
  - 将所有列导出为 NumPy 数组。字符串列直接返回切片，数值列经过 `backend.to_numpy()` 转换。

EntityPoolView

- 由 `EntityPool.query(..., return_='view')` 返回。
- 保存 `(pool, idxs)`，只有在访问某列时才会读取底层数据。
- 支持：
  - `view['hp']` / `view.get_col('hp')` 读取列数据（返回 ndarray，对列再做 fancy indexing 将产生拷贝）。
  - `view['hp'] = vals`（按索引对被选实体批量赋值）。
  - `view.assign({'hp': arr, 'type': val})`（多列批量赋值）。

示例：

```python
view = engine.query(lambda p: p.get_attr('type') == 1, return_='view')
view['hp'] = 10.0
view.assign({'type': 9, 'hp': np.arange(len(view))})
```


Relation / RelationView / SparseRelationEdges / DenseRelation

Relation（稀疏边表）

构造：

```
Relation(name: str, capacity: int = 1024, attr_dtypes: Optional[dict] = None, backend: str = 'numpy')
```

- name: 关系名称。
- capacity: 初始容量（默认 1024）。
- attr_dtypes: 边属性字典，键为属性名，值为 dtype。
- backend: 计算后端，支持 'numpy'（默认）/ 'torch' / 'mlx' / 'jax' / 'auto'。

常用方法（用户层）

- add(src_uids, dst_uids, **attrs) -> np.ndarray[int]
  - 批量添加边，返回新边在边表中的下标数组。

- remove_by_indices(idxs)
  - 按边的下标删除（会压缩边表）。

- query(src_uid=None, dst_uid=None, edge_pred=None, return_='indices')
  - 按条件查询边，return_ 支持 'indices' / 'mask' / 'view' / 'relation'（relation 返回子 Relation 的拷贝）。

- index(...)
  - 轻量索引层，只返回 indices/mask。

- set_attr(name, idx, values, *, backend='auto')
  - 对单列按边索引赋值（支持标量/一维数组）。

- assign(idx, values_by_attr: dict, *, backend='auto')

- where_set(...)
  - 按谓词对匹配边批量赋值，返回命中边的 idx 数组。

- row_sum / col_sum / row_count / col_count
  - 基于 uid->pos 映射对边属性求和/计数，适用于快速聚合。

RelationView

- 由 `Relation.query(..., return_='view')` 返回，保存 `(relation, edge_idxs)`。
- 支持：
  - `view['dst_uid']` / `view.get_col('weight')` 读取列数据（只读取被选边）。
  - `view['weight'] = vals` / `view.assign({...})` 对被选边批量赋值。

SparseRelationEdges

- 轻量只读结构：保存 src_uid/dst_uid/(value) 三列，提供 index/query 等工具函数用于快速分析/过滤（例如构建 uid->pos 的映射）。

DenseRelation

- 用二维 ndarray 表示矩阵关系，支持 `index(rows, cols, return_='indices'|'mask')` 与 `query(rows, cols, pred, return_='dense'|'coords'|'indices')`。


工具函数

- to_indices(idx, size) -> np.ndarray[int]
  - 将多种索引（None/int/slice/array/bool mask）标准化为整型下标数组。

- to_mask(sel, size) -> np.ndarray[bool]
  - 将索引或布尔掩码标准化为布尔掩码。

- dense_to_relation(mat, src_pool, dst_pool=None, rel_name='relation', attr_name='weight', threshold=0.0) -> Relation
  - 将稠密矩阵转换为 `Relation`。


示例汇总

- 批量创建实体并更新：

```python
engine = RECS(128, {'hp': 'float64', 'type': 'int32'})
idxs = engine.add(3, hp=[10, 20, 30], type=1)
engine.set_attr('hp', idxs, np.array([11, 22, 33]))
```

- 按谓词批量设置属性：

```python
engine.where_set(lambda p: p.get_attr('hp') < 20, {'hp': 20.0, 'type': 0})
```

- 构造 relation 并查询：

```python
rel = engine.dense_matrix_to_relation(np.array([[0,1],[2,0]]), relation_name='r', attr_name='w')
edge_idxs = rel.query(src_uid=engine.uid_array()[0], return_='indices')
view = rel.query(src_uid=engine.uid_array()[0], return_='view')
view['w'] = 100.0
```

---

版本与兼容性

- 该模块以 `RECS` 作为唯一对外主入口，便于统一维护。
- 对外暴露的类型与接口旨在尽量贴合 NumPy 的索引/赋值语义，降低学习成本。
- 多后端架构：默认 NumPy 后端零退化兼容；可选 PyTorch/MLX/JAX 后端按需加载。


## 后端工具函数（Backend Utilities）

### recs.backends

```python
from recs.backends import get_backend, list_backends
```

- **list_backends() -> list[str]**
  - 返回当前所有已注册（可用）的后端名称列表，例如 `['numpy', 'torch', 'mlx', 'jax']`。

- **get_backend(name: str = 'numpy') -> BackendBase**
  - 获取指定名称的后端实例。
  - `name='auto'` 时自动选择最优可用后端（优先级：mlx > torch > jax > numpy）。
  - 当指定后端未安装/不可用时抛出 `ValueError`。

### BackendBase 接口

`BackendBase` 是定义在 `recs/backends/base.py` 中的抽象基类，所有后端共享约 28 个数组操作原语：

```python
# 元信息
backend.name     # → 'numpy' | 'torch' | 'mlx' | 'jax'
backend.device   # → 'cpu' | 'mps'

# 数组创建
backend.empty(shape, dtype)
backend.zeros(shape, dtype)
backend.ones(shape, dtype)
backend.full(shape, fill_value, dtype)
backend.arange(start, stop, step, dtype)
backend.array(data, dtype)

# 数组操作
backend.where(condition, x, y)
backend.concatenate(arrays, axis=0)
backend.copy(arr)
backend.asarray(obj, dtype)
backend.astype(arr, dtype)        # 跨后端类型转换

# 索引与查询
backend.nonzero(arr)
backend.boolean_mask(arr, mask)
backend.isin(elements, test_elements)

# 聚合
backend.bincount(indices, weights, minlength)
backend.add_at(output, indices, values)
backend.sum(arr, axis)
backend.all(arr, axis)

# 排序与唯一
backend.argsort(arr, kind='stable')
backend.unique(arr)

# 类型转换
backend.to_numpy(arr)             # → np.ndarray
backend.from_numpy(np_arr)        # → backend array

# 设备管理
backend.to_device(arr, device)    # 'cpu' / 'gpu' / 'mps'
backend.cpu(arr)
backend.gpu(arr)
```


