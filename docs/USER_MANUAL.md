# 用户手册（User Manual）

目录

- 快速开始
  - 环境与依赖
  - 运行示例
- 概览（概念）
- 面向用户的优势（高阶视角）
- 公共 API 快速参考（简明）
  - RECS（高层）
  - EntityPool / EntityPoolView
  - Relation / RelationView / SparseRelationEdges / DenseRelation
  - 主要工具函数（to_indices / to_mask / dense_to_relation）
- 常用示例（快速上手）
  - 创建并填充实体池
  - 使用索引/查询
  - 通过 view 做批量赋值（index then assign）
  - 从矩阵构造关系
- 调试与故障排查
- 推荐用法：统一 index / query
- 参考：更详细的 API 说明见 docs/API_REFERENCE.md

快速开始

环境与依赖

- Python 3.8+
- NumPy 1.20+（必需，始终作为默认后端）

可选加速后端（按需安装）：

```bash
# PyTorch 后端 — 支持 MPS (Apple Silicon GPU) / CUDA
pip install "recs[torch]"

# MLX 后端 — Apple Silicon 统一内存，零拷贝
pip install "recs[mlx]"

# JAX 后端 — JIT 编译（仅只读操作）
pip install "recs[jax]"

# 或直接安装对应包
pip install torch mlx jax
```

在 `relation-entity-component-system` 中，项目已经包含纯 Python 代码和示例。若你把本目录作为包使用，请确保你的环境已安装 NumPy：

```powershell
python -m pip install numpy
```

运行示例

仓库中包含 `demos/demo1.py` 和 `demos/demo2.py`，运行它们来观看基本用法：

```powershell
python demos\demo1.py
python demos\demo2.py
```





# 概览（概念）

- 实体（Entity）：由 `EntityPool` 管理，具有唯一的 uid（字段 `i`）与启用标志 `o`。实体数据以列（每列为 ndarray）的形式存储。
- 组件/属性（Attributes）：每个属性是 `EntityPool.d` 或 `Relation.d` 中的一列 ndarray，可动态添加。
- 关系（Relation）：边表，以 `src_uid` / `dst_uid` 为主键，其他属性按列式保存。既支持稠密矩阵（`DenseRelation`）也支持稀疏边表（`Relation` / `SparseRelationEdges`）。

面向用户的优势（高阶视角）

这里尽量少涉及底层实现，直接描述用户在使用层面能得到的好处：

- NumPy 风格的向量化体验：可以按列写出一次性批量计算与更新，简洁且高效，避免对每个实体/边写显式的 Python 循环。
- 先索引再查询的两步语义（index -> query）：方便组合复杂条件，再决定是否材料化或返回 view（延迟取列）。
- 视图（view）语义：`EntityPoolView` / `RelationView` 保存索引并延迟获取列，避免不必要的大规模拷贝。
- 直接对查询结果赋值：支持像 NumPy 那样先索引再对索引位置赋值（例如：`view['hp'] = vals` 或 `engine.where_set(...)`）。
- UID 稳定：关系中保存 uid 而不是内存索引，实体在内部移动或压缩时关系仍一致。
- 同时兼顾稠密/稀疏场景：提供稠密矩阵与稀疏边表的互转与并用，用户不必关心底层存储即可完成常见任务。
- 与科学计算生态兼容：数据存放为 NumPy ndarray，易与 NumPy/SciPy/PyTorch 等工具链衔接。

公共 API 快速参考（简明）

本节只列出用户常用的函数/方法和它们的快速语义说明。更细的参数、返回值、例子见 `docs/API_REFERENCE.md`。

RECS（高层）

- 构造：`RECS(capacity, attr_dtypes)` — 创建引擎并预分配字段（attr_dtypes 为字典）。
- 常用方法：
  - `add(n=1, **kwargs)` — 批量添加实体；可通过 kwargs 设置各属性（支持广播）。
  - `remove(idx)` / `enable(idx)` — 逻辑禁用/启用行（修改 `o` 字段）。
  - `add_attribute(name, dtype, default=0)` — 动态添加列并用 default 初始化。
  - `get_attr(name)` / `get_attrs_view(names)` — 获取列视图（只包含前 size 项）。
  - `index(pred, return_='indices')` / `query(pred, return_='view')` — 索引/查询接口（见下文）。
  - `set_attr(name, idx, values)` / `assign(idx, values_by_attr)` / `where_set(pred, values_by_attr)` — 对查询或索引位置批量赋值（支持标量广播与与选中行数长度相匹配的数组）。
  - `dense_matrix_to_relation(mat, dst_pool=None, relation_name='relation', attr_name='weight', threshold=0.0)` — 将稠密矩阵转换为 Relation（返回新的 Relation）。

EntityPool / EntityPoolView

- `EntityPool`：底层表格化存储，常用方法：`add`、`remove`、`enable`、`add_attribute`、`get_attr`、`query`、`index`、`set_attr`、`assign`、`where_set`。
- `EntityPoolView`：由 `query(return_='view')` 返回，轻量保存 `(pool, idxs)`，支持 `view['col']`、`view.get_col('col')`、`view['col']=vals`、`view.assign({...})`。

Relation / RelationView / SparseRelationEdges / DenseRelation

- `Relation`：稀疏边表实现，提供 `add(src_uids, dst_uids, **attrs)`、`query(...)`/`index(...)`、`set_attr`/`assign`/`where_set`、`row_sum`/`col_sum` 等。
- `RelationView`：轻量视图，支持像表一样读取列并对选中边批量赋值（`view['weight'] = vals` / `view.assign({...})`）。
- `SparseRelationEdges`：用于只读场景下的稀疏边表索引/查询工具（构建 uid->pos 映射并执行快速筛选）。
- `DenseRelation`：用于矩阵化关系，支持类似 `mat[rows, cols]` 的索引/查询。

常用示例（快速上手）

创建引擎并添加实体：

```python
from recs import RECS
engine = RECS(64, {'hp': 'float64', 'type': 'int32'})
# 添加实体：返回新实体在 pool 中的下标范围
idxs = engine.add(2, hp=[50.0, 80.0], type=[1, 2])
# 禁用第 0 个实体（逻辑删除）
engine.remove(0)
# 重新启用
engine.enable(0)
```

使用索引/查询（index -> query -> 操作）：

```python
# 计算索引（轻量）
idx = engine.index(lambda p: p.get_attr('hp') > 50, return_='indices')
# 也可以直接拿 view（延迟取列）
view = engine.query(lambda p: p.get_attr('hp') > 50, return_='view')
print(view['hp'])  # 只有在取列时才会访问 d['hp']
# 材料化记录（会拷贝）
records = engine.query(lambda p: p.get_attr('hp') > 50, return_='records')
```

通过 view 做批量赋值（NumPy 风格的先索引再赋值）：

```python
# 1) 使用 view 的 __setitem__（推荐）
view = engine.query(lambda p: p.get_attr('type') == 1, return_='view')
# 标量广播
view['hp'] = 10.0
# 或者按实体个数传入同长度数组
view['hp'] = np.array([11.0, 12.0, 13.0])

# 2) 通过 engine.where_set（按谓词直接赋值）
engine.where_set(lambda p: p.get_attr('hp') < 20, {'type': 9, 'hp': 20.0})
```

从矩阵构造关系（稠密 -> 边表）：

```python
import numpy as np
mat = np.array([[0, 1.0],[0.5, 0]])
rel = engine.dense_matrix_to_relation(mat, relation_name='connects', attr_name='weight')
print(rel.d['src_uid'][:rel.size])
print(rel.d['dst_uid'][:rel.size])
```

调试与故障排查

- "Length mismatch" 错误：检查向量属性长度是否与要赋值的实体/边数一致，或是否为可广播的标量。
- 索引越界：确保传入的下标或数组中的值小于当前 `size`；用 `len(engine)` 或 `engine.size` 检查当前大小。
- 空结果：当 `query` 命中 0 个实体/边时，`indices` 为空数组，`view` 是长度 0 的视图，`records` 是空字典/数组集合。

推荐用法：统一 index / query

为了避免用户同时面对“实体表/关系表 + 稠密/稀疏”的多套接口，本项目推荐只记住两类操作：

- 索引（index）：只返回位置（indices/mask），便于组合多个条件，代价低。
- 查询（query）：在索引基础上返回想要的数据形态（view/copy/relation），根据需要选择是否材料化。

示例（EntityPool / RECS）：

```python
# 只获取下标（轻量）
idx = engine.index(lambda p: p.get_attr('hp') > 50, return_='indices')

# 获取 view（延迟取列）
view = engine.query(lambda p: p.get_attr('hp') > 50, return_='view')

# 材料化为 records（拷贝）
records = engine.query(lambda p: p.get_attr('hp') > 50, return_='records')
```

示例（Relation）：

```python
edge_idx = rel.index(src_uid=[uid1, uid2], return_='indices')
edge_view = rel.query(src_uid=uid1, edge_pred=(rel.get_attr('amount') > 1000), return_='view')
print(edge_view['dst_uid'])
sub_rel = rel.query(src_uid=uid1, edge_pred=(rel.get_attr('amount') > 1000), return_='relation')
```

参考：更详细的 API 说明见 `docs/API_REFERENCE.md`（新增）

# 多后端计算（Multi-Backend）

RECS 支持在 NumPy（默认）、PyTorch、MLX、JAX 等多个计算后端上透明运行。

### 选择后端

创建引擎时传入 `backend` 参数即可：

```python
# NumPy 后端（默认，不传 backend 时自动使用）
import numpy as np
from recs import RECS

r = RECS(64, {'x': float})          # 默认 numpy
r = RECS(64, {'x': float}, backend='numpy')
```

```python
# PyTorch 后端（需安装 torch）
r = RECS(64, {'x': float}, backend='torch')
# 数据自动存为 torch.Tensor（MPS 设备上）
print(r.d['x'].device)  # → mps:0
```

```python
# MLX 后端（需安装 mlx，仅 Apple Silicon）
r = RECS(64, {'x': float}, backend='mlx')
# 数据自动存为 mx.array（统一内存）
```

```python
# JAX 后端（需安装 jax，仅只读操作）
from recs.backends import get_backend
b = get_backend('jax')
arr = b.zeros(10, dtype=float)
```

```python
# auto 模式：自动选择最优可用后端
r = RECS(64, {'x': float}, backend='auto')
```

### 检查可用后端

```python
from recs.backends import list_backends, get_backend

print(list_backends())           # → ['numpy', 'torch', 'mlx', 'jax']
b = get_backend('numpy')
print(b.name, b.device)          # → numpy cpu
```

### 跨后端数据导出

```python
# 无论使用哪个后端，都可以导出为 NumPy 数组
r = RECS(64, {'x': float}, backend='torch')
r.add(3, x=[1.0, 2.0, 3.0])
out = r.to_numpy()               # → {'x': np.ndarray}
print(out['x'])                  # → [1. 2. 3.]
```

### 后端兼容性说明

| 后端 | 状态 | 设备 | 字符串列 | 写入 | 备注 |
|------|------|------|----------|------|------|
| numpy | ✅ 生产就绪 | CPU | ✅ 原生支持 | ✅ 完整 | 默认，零依赖 |
| torch | ✅ 生产就绪 | MPS/CUDA | ✅ 自动路由 NumPy | ✅ 完整 | 推荐 GPU 加速 |
| mlx | ✅ 生产就绪 | MPS | ✅ 自动路由 NumPy | ✅ 完整 | Apple Silicon 优化 |
| jax | ⚠️ 只读 | CPU | ❌ | ❌ 不可变数组 | 仅后端级查询 |
| tensorflow | ❌ 不可用 | — | — | — | Python 3.14 不兼容 |
