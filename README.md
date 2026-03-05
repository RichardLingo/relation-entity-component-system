ECS / SoA ECSEngine
=================

概览
----
本仓库实现了一个轻量级的 SoA（Structure of Arrays）风格的引擎 `ECSEngine`，适合作为表格型实体数据的存储与批量处理基础。

核心设计要点（目前实现）
----------------------
- 存储布局：采用 SoA，每个属性（field）对应一维 NumPy 数组，内部存储字典命名为 `d`（key->ndarray）。
- 批量操作优先：所有实体操作以批量为主，`add(n, **kwargs)` 是主入口（单个添加为 n=1 的特例）。
- 启用标识：使用属性 `o`（on 的简写）作为唯一的启用/激活掩码（布尔数组）。不再使用额外的 `is_active`。
- 唯一 id：使用属性 `i` 保存实体 uid（np.int64），引擎维护自增计数器 `_i_counter`。
- 扩容策略：以指数增长（*2）扩容，扩容时对新增区域做向量化初始化（`np.arange` / `np.full` / 广播），减少 Python 循环。
- 性能原则：优先使用 NumPy 向量化、C-contiguous arrays 和切片赋值，避免每元素的 Python 循环。

主要 API
---------
类：`ECSEngine`

构造：
- `ECSEngine(capacity: int, attr_dtypes: dict)`
  - `capacity`：初始物理容量（列长度）。
  - `attr_dtypes`：用户自定义字段 -> numpy dtype 映射（例如 `{'position': np.float32}`）。
  - 引擎会自动添加并管理两个必备字段：
    - `i`：uid（np.int64），自增分配
    - `o`：on（bool），表示启用/激活

核心方法：
- `add(n=1, **kwargs) -> np.ndarray`
  - 批量添加 `n` 个实体，返回新增实体的下标数组（np.ndarray of int）。
  - 默认新增实体 `o=True`（即新增即启用）；如果不希望启用，可在 `kwargs` 中传入 `o=False`，或随后调用 `remove(indices)`。
  - `kwargs` 可为标量（会广播）或长度为 `n` 的数组。

- `batch_add(n, **kwargs)`
  - 兼容别名，等同 `add(n, **kwargs)`。

- `add_attribute(name, dtype, default=0)`
  - 为当前容量添加新字段（列），用 `default` 初始化。

- `get_attr(name) -> np.ndarray`
  - 返回字段 `name` 的视图（只包含前 `size` 项）。

- `active_mask -> np.ndarray(bool)`
  - 返回长度为 `size` 的布尔掩码，表示哪些实体启用（基于 `d['o']`）。

- `get_active_indices() -> np.ndarray`
  - 返回当前启用实体的下标数组。

- `remove(idx_or_mask_or_slice)` / `enable(idx_or_mask_or_slice)`
  - 支持多种索引类型：单个整数、整型数组、slice、或布尔掩码（长度必须等于当前 `size`）。
  - 向量化设置 `d['o']` 为 False/True。

内部/行为说明
-------------
- 内部存储字典 `d`：键为字段名，值为长度为 `capacity` 的 NumPy ndarray（C-contiguous）。
- `size`：表示当前已分配的实体数（逻辑行数）。物理数组长度为 `capacity`。
- 扩容：当 `size + n > capacity` 时，自动扩容到 `max(capacity*2, size+n)`，并对新增尾部批量初始化：
  - `i` 用连续自增 uid 初始化；
  - `o` 初始化为 False；
  - 其它字段按 `_defaults` 或类型选择初始化（数值类型为 0，非数值为 None）。

使用示例（demo1）
-----------------
下面示例在 `Projects/ECS/demos/demo1.py` 中已有实现：

```python
import numpy as np
from ecs.engine import ECSEngine

engine = ECSEngine(capacity=4, attr_dtypes={'position': np.float32, 'velocity': np.float32})

# 添加单个实体（n=1，返回 array([idx])）
idx = engine.add(1, position=1.0, velocity=2.0)

# 批量添加 3 个实体（标量和序列混合）
idxs = engine.add(3, position=[0.1,0.2,0.3], velocity=5.0)

# 添加新字段
engine.add_attribute('health', dtype=np.int32, default=100)

# 禁用 / 启用（支持 int/list/slice/bool-mask）
engine.remove(idx)          # 单个或数组
engine.enable(slice(0, 10))

# 读取视图
positions = engine.get_attr('position')
active = engine.active_mask
active_indices = engine.get_active_indices()
```

运行 demo 的提示
----------------
- 以包方式运行或确保 Python 能找到 `ecs` 包：将包含 `Projects/ECS` 的上级目录加入 `PYTHONPATH`，例如（Windows PowerShell）：

```powershell
$env:PYTHONPATH = "C:\Users\Ethan\CoreFiles\ProjectsFile\ComplexSystemLab\ComplexSystemLab"
python .\Projects\ECS\demos\demo1.py
```

集成建议
--------
- 目前 `d` 是内部存储细节；推荐外部模块通过 `get_attr()`、`add_attribute()`、`add()`、`remove()`、`enable()` 等 API 操作，而不是直接读取 `d`。
- 若未来需要 PyTorch/GPU 整合，可考虑把某些字段存成 `torch.Tensor`，或在需要计算时用 `torch.from_numpy()` 批量转换。
- 如果准备把本引擎打包为 pip 包：添加 `pyproject.toml`/`setup.cfg`，`__init__.py` 导出 `ECSEngine`，并添加测试与 CI。

文档 / 文档说明
----------------
本项目的用户与开发文档存放在 `docs/` 目录中，已包含下列文件（相对于 `Projects/ECS` 目录）：

- [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — 开发者指南：架构说明、主要数据结构、扩容策略、示例代码与故障排查。
- [USER_MANUAL.md](docs/USER_MANUAL.md) — 用户手册：快速上手、依赖说明、API 参考与常见示例。
- [PRD.md](docs/PRD.md) — 产品需求文档：目标、用例、功能/非功能需求与路线图。

你可以直接在本地打开或在编辑器中查看这些文档，例如在 Windows PowerShell 中：

```powershell
# 从仓库根运行（示例）
code .\Projects\ECS\docs\DEVELOPER_GUIDE.md   # 或使用你喜欢的编辑器
notepad .\Projects\ECS\docs\USER_MANUAL.md
```

请参阅 docs 中的文件以获取更详细的使用和开发说明。

更多相关资料
------------
- [图数据结构与算法综述（notes/graph_structures_and_algorithms.md）](../../../../../../CS_notebook/Projects/ECS/notes/graph_structures_and_algorithms.md)

注意事项
--------
- 布尔掩码索引会返回拷贝，谨慎在大数据上频繁使用布尔索引。优先使用切片或整数索引组合以避免不必要的拷贝。
- 扩容会对每列进行复制；建议在性能关键路径中预分配足够容量或使用 `reserve()`。

联系方式
--------
如需我把项目封装为可安装的包、补充单元测试或做 PyTorch 集成示例，我可以继续把这些工作做完。
