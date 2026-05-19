# 开发者指南（Developer Guide）

目录

## 二维数组操作规范（实体表 / 关系表，稠密 / 稀疏）

目标：为 RECS 中两类二维数组（实体表、关系数组）在两种数据结构形式（稠密形式、稀疏形式）上的各种操作建立统一约定，提供类似 NumPy 的语法糖，并在默认缺省参数下由后台自动判断并路由到合适的实现；当用户显式传入参数时按参数指定的数据结构类型调用对应算法。

设计原则：
- 统一外部 API：尽量只暴露一套方法名（例如 `index`, `query`, `take`, `where`, `aggregate`, `apply` 等），保持用户接口简单且符合 NumPy 使用习惯。
- 后端多实现：对于两类二维数组（实体表、关系数组）以及两种数据结构形式（稠密、稀疏），应支持四种后端实现的组合：
  - 实体表（dense）
  - 实体表（sparse）
  - 关系数组（dense）
  - 关系数组（sparse）
 这些实现可以放在同一类中或分离为独立模块，但接口必须保持一致以便路由切换与测试。
- 自动路由：默认 `backend='auto'`（缺省），由系统在运行时基于数组内存布局与稀疏指标自动选择上述四类中最合适的一种实现以获得最佳性能。
- 显式覆盖：允许用户通过参数强制选择 `backend='dense'` 或 `backend='sparse'` 来使用指定实现，主要用于调试或性能对比。
- 语法糖与兼容性：提供 NumPy 风格的索引/广播/掩码语法糖（例如布尔掩码、花式索引、切片、广播运算），并确保行为在 `EntityPool` 与 `Relation` 间一致。

推荐 API 约定（示例）：
- 通用签名（实体表/关系表）:

```
obj.operation(*args, backend='auto', /, **kwargs)
```

- 说明：
  - `obj`：可为 `EntityPool`、`ECSEngine.d`（实体表视图）、或 `Relation`。
  - `operation`：例如 `take`, `where`, `apply`, `aggregate`, `join`, `to_dense`, `to_sparse` 等。
  # 开发者指南（Developer Guide）

  本文档以软件工程视角全面描述 ECS 核心库的设计约定、开发流程、编码规范与运行时契约，旨在为新增贡献者与维护者提供一致参考。

  目录

  - 概要
  - 快速开始（Quickstart）
  - 架构概览
  - 数据模型与主要类型
  - 公共 API 与约定
  - 二维数组操作规范（实体表 / 关系表，稠密 / 稀疏）
  - 编码规范（风格、文档、错误处理）
  - 测试策略与 CI
  - 性能与扩容建议
  - 向后兼容与迁移策略
  - 贡献者指南（Pull Request、分支、代码审查）
  - 发布与变更日志
  - 安全、许可与联系方式

  概要

  RECS 是一个轻量级 relation-entity-component-system 核心库。实现目标为：易用、可嵌入、便于向量化（NumPy 优化路径），同时对外提供向后兼容的 API 以便现有用户平滑过渡。

  快速开始（Quickstart）

  - 安装依赖：`pnpm install`（前端）或 `pip install -r requirements.txt`（若有 Python 依赖）。
  - 生成项目树（若需要）：`pnpm gen:tree`。
  - 运行示例：参见 `demos/` 中的脚本，如 `demos/demo1.py`。

  架构概览

  - 存储策略：SoA（Structure of Arrays），每个属性为独立的 ndarray/列，方便列向量化操作。
  - 主要模块：`ecs` 包含 `ecsengine.py`、`EntityPool`、`Relation`；`demos/` 提供使用示例。
  - 扩展点：后端实现（dense/sparse）、查询/索引语义、序列化接口。

  数据模型与主要类型

  - `EntityPool`：实体集合管理器，关键字段 `d`（列字典）、`i`（uid 列）、`o`（启用标记）、`capacity`、`size` 等。
  - `Relation`：边/关系表，列式存储 `src_uid`,`dst_uid` 与其它边属性。
  - `ECSEngine`：兼容包装器，暴露常用代理方法并管理多个实体池/关系。

  公共 API 与约定

  - 对外优先暴露的核心方法集合：`index`, `query`, `add`, `remove`, `add_attribute`, `reserve`, `take`, `where`, `aggregate`。
  - 所有对外 API 的共识：输入参数支持 NumPy 风格（切片、布尔掩码、花式索引）；返回类型通过 `return_` 或显式 API 指定（`indices|mask|view|copy|relation`）。
  - 参数 `backend='auto'|'dense'|'sparse'`：缺省为 `'auto'`，由系统根据数据稀疏度与内存布局选择实现；用户可显式指定以覆盖自动路由。

  二维数组操作规范（实体表 / 关系表，稠密 / 稀疏）

  目标：为 RECS 中两类二维数组（实体表、关系数组）在两种数据结构形式（稠密形式、稀疏形式）上的各种操作建立统一约定，提供 NumPy 风格语法糖，并在默认缺省参数下由后台自动判断并路由到合适实现；当用户显式传入参数时按参数指定的数据结构类型调用对应算法。

  - 要点概览：
  - 统一 API：对用户只暴露一套方法名（`index/query/take/where/aggregate/apply` 等），保持一致性。
  - 后端实现：每个操作应覆盖两类二维数组（实体表 / 关系数组）与两种数据结构形式（稠密 / 稀疏）的组合（共四种后端实现）。允许部分操作仅实现其中的子集，但必须在文档中明确说明并提供退路或合理的 `auto` 路由策略。
  - 自动路由策略：`backend='auto'` 时基于稀疏度阈值、位图可用性与内存布局选择实现；可通过配置覆写阈值。
  - 显式覆盖：允许用户传 `backend='dense'` 或 `backend='sparse'` 强制使用对应实现。
  - 语法糖：兼容 NumPy 的布尔掩码 / 花式索引 / 切片 / 广播 语法，保证 `EntityPool` 与 `Relation` 行为一致。

  接口范式示例：

  ```python
  # 通用签名
  res = obj.take(indices, backend='auto', return_='view')

  # 显式选择稠密/稀疏后端
  res_d = obj.where(pred, backend='dense')
  res_s = rel.query(src_uid=uids, backend='sparse')
  ```

  实现建议（工程级）:
  - 分层实现：参数解析层（统一解析各种索引/掩码）→ 路由层（选择 dense/sparse）→ 后端实现层。
  - 自动检测：建议基于非零比例（非默认值 / 总大小）和位图/索引可用性判定；默认稀疏阈值可在全局配置中调整。
  - 性能实现：稠密路径使用 NumPy 向量化；稀疏路径可使用 CSR/COO 或位图集合以减少遍历与内存占用。
  - 语义一致性：不同后端必须保证结果语义一致；单元测试验证 `dense`/`sparse`/`auto` 三者的一致性。

  编码规范（风格、文档、错误处理）

  - 语言风格：遵循 PEP8；类型注解优先使用（Python 3.9+ 风格）。
  - 文档注释：Python 文件使用 Google 风格文档字符串并补充中文说明（仓库约定）。每个公共函数/类/模块必须包含用途、参数、返回值与示例。
  - 错误处理：外部 API 抛出明确异常类型（如 `ValueError`, `IndexError`, `ECSConfigurationError`），并在文档中列出可能抛出的异常。
  - 日志：使用标准 `logging`，并在关键路径（扩容、路由选择、重建索引）记录 DEBUG/INFO 级别日志。

  测试策略与 CI

  - 单元测试：为每个模块建立单元测试，覆盖常用场景与边界条件；特别要求对 `backend='dense'|'sparse'|'auto'` 三种路径进行一致性测试。
  - 集成/烟雾测试：`demos/` 中的脚本应作为 smoke-tests 在 CI 中运行，验证核心接口不回归。
  - CI：建议在 PR 中运行 `pnpm lint`（若前端）和 Python 单元测试（`pytest`），并在主分支触发 release tests。

  性能与扩容建议

  - 扩容策略：默认按需倍增（new_capacity = max(current*2, needed)），避免频繁 realloc。
  - 预分配：在批量插入前使用 `reserve(min_capacity)` 减少复制次数。
  - 向量化优先：优先使用 NumPy 批量操作替代 Python 循环，必要时使用 C/Numba/Cython 加速。

  向后兼容与迁移策略

  - 兼容层：`ECSEngine` 提供代理方法，将老接口映射到新约定；任何移除的老接口必须在文档中标注弃用周期。
  - 迁移文档：为重大接口变更提供迁移示例代码段（old → new）。

  贡献者指南（PR、分支、代码审查）

  - 分支策略：使用 feature/bugfix/release 命名法；对外 PR 目标为 `main` 或 `develop`（依项目流程）。
  - 提交规范：每次 PR 包含变更说明、测试说明、以及必要的 benchmark 结果（如涉及性能路径）。
  - 代码审查：至少一名 reviewer 批准；修改涉及公共 API 时需在 `docs/CHANGELOG.md` 记录影响。

  发布与变更日志

  - 在 `docs/CHANGELOG.md` 中记录每次对外可见变更（新特性、兼容性变更、bugfix）。
  - 发布流程：从 `main` 创建 release tag，并把变更记录整理在 release notes 中。

  安全、许可与联系方式

  - 代码遵循仓库根部的 LICENSE。若引入第三方依赖请审查其许可证兼容性。
  - 报告安全问题请联系仓库维护者或在内部通道提交安全报告。

  示例与参考

  - 使用示例：参见 `demos/`。
  - 主要实现：参见 `ecs/ecsengine.py`、`ecs/__init__.py`。

  附录：下一步建议

  - 将本指南作为 PR 模板的一部分，要求贡献者在 PR 描述中对照本指南填写影响项。
  - 把“二维数组操作规范”拆成可执行的检测清单并添加到 `docs/CHANGELOG.md` 的变更示例中。

  赋值操作补充规范（索引/查询后写入）

  为保证“统一 API + NumPy 语义”，在索引与查询之外，推荐补齐以下写入操作：

  - 单列按索引赋值（基础原语）：`obj.set_attr(name, idx, values, backend='auto')`
  - 多列按索引批量赋值：`obj.assign(idx, values_by_attr, backend='auto')`
  - 按查询条件直接赋值：`obj.where_set(..., values_by_attr, backend='auto')`
  - 视图赋值：`view['attr'] = values`、`view.set_attr(...)`、`view.assign(...)`

  统一语义约定：

  - `idx` 支持 int/slice/布尔掩码/整型数组；行为与 `index/query` 一致。
  - `values` 支持标量广播或与命中数量一致的一维数组。
  - `backend` 统一支持 `'auto'|'dense'|'sparse'`；未实现的后端应至少保持接口兼容并在文档说明当前等价路径。
  - `view` 为“可写选择器”：读可延迟取列，写必须回写到底层对象（`EntityPool`/`Relation`）。

  当前实现覆盖（2026-02-19）：

  - `EntityPool`：已支持 `set_attr/assign/where_set`。
  - `Relation`：已支持 `set_attr/assign/where_set`。
  - `EntityPoolView`：已支持 `view['col']=...`、`set_attr`、`assign`。
  - `RelationView`：已支持 `view['col']=...`、`set_attr`、`assign`。
