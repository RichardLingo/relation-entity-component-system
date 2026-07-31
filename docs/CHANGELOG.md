# 变更日志

本文件记录每个版本发生了什么。若某个版本包含用户写法变化、默认行为变化或破坏性变更，请同时更新：

- [迁移指南](MIGRATION_GUIDE.md)：说明用户代码如何从旧写法迁移到新写法。
- [API 弃用策略](DEPRECATION_POLICY.md)：说明废弃、兼容期和移除规则。
- [发布检查清单](RELEASE_CHECKLIST.md)：发版前确认文档、warning、demo 和测试已同步。

## 未发布

### Added

- 

### Changed

- 

### Deprecated

- 

### Removed

- 

### Fixed

- 

### Migration

- 如本版本包含用户可见写法变化，在此链接到 `docs/MIGRATION_GUIDE.md` 中对应条目。

## 2026-06-23

### 新增：多后端计算架构（v0.2）

**核心功能：** 在 NumPy 默认后端之外，支持 PyTorch、MLX、JAX 作为可选计算后端，EntityPool/Relation 可在不同后端上透明运行。

#### 新增文件

- `recs/backends/__init__.py` — 后端注册表，支持惰性注册与自动检测
- `recs/backends/base.py` — `BackendBase` 抽象基类，定义约 28 个数组操作原语
- `recs/backends/numpy_backend.py` — `NumpyBackend` 实现（默认，零额外依赖）
- `recs/backends/torch_backend.py` — `TorchBackend` 实现（支持 MPS/CUDA）
- `recs/backends/mlx_backend.py` — `MLXBackend` 实现（Apple Silicon 统一内存）
- `recs/backends/jax_backend.py` — `JAXBackend` 实现（JIT 编译，仅只读操作）
- `recs/backends/tf_backend.py` — `TFBackend` 实现（TensorFlow，Python 3.14 暂不可用）

#### 修改文件

- `ecs/ecsengine.py`：
  - `EntityPool.__init__` 新增 `backend='numpy'` 参数
  - `Relation.__init__` 新增 `backend='numpy'` 参数
  - `RECS.__init__` 新增 `backend='numpy'` 参数
  - 所有 `np.*` 调用委托为 `self.backend.*`
  - 字符串列强制使用 NumPy（通过 `_string_columns` 集合与 `_is_string_value` 检测）
  - `EntityPool` 新增 `to_numpy()` 方法导出为 NumPy 数组
  - `EntityPool` 新增 `is_active(idx)` 方法
  - `RECS` 新增 `is_active(idx)`、`disable(idx)` 代理方法
  - `RECS` 暴露 `self.backend` 引用

#### 新测试文件

- `test_multibackend.py` — 多后端集成测试
- `tests/comprehensive_test.py` — 综合边缘测试套件（8 组测试，覆盖 6 个后端）

#### 已知限制

- TensorFlow 后端因 Python 3.14 不兼容暂不可用
- JAX 后端仅支持只读操作（JAX 数组不可变）
- MLX 不支持布尔索引（`boolean_mask` 回退为整数索引）
- MLX 不支持 `add_at` 原地操作
- PyTorch MPS 不支持 float64（自动降级为 float32）
- MLX GPU scatter 不支持 int64（自动降级为 int32）

## 2026-02-19

### 新增
- 为实体表 `EntityPool` 新增统一赋值接口：`set_attr(name, idx, values, backend='auto')`、`assign(idx, values_by_attr, backend='auto')`、`where_set(pred, values_by_attr, backend='auto')`。
- 为关系表 `Relation` 新增统一赋值接口：`set_attr(name, idx, values, backend='auto')`、`assign(idx, values_by_attr, backend='auto')`、`where_set(..., values_by_attr, backend='auto')`。
- 为 `EntityPoolView` 与 `RelationView` 新增可写语义：支持 `view['attr'] = values`、`set_attr(...)`、`assign(...)`，可直接回写底层对象。
- 新增统一后端参数校验：`backend` 允许 `'auto'|'dense'|'sparse'`，用于与开发者指南的统一 API 约定保持一致。

### 文档
- 在 `docs/DEVELOPER_GUIDE.md` 增补“赋值操作补充规范（索引/查询后写入）”，明确查询后赋值、批量赋值与 view 赋值语义。

### 示例
- 在 `demos/demo2.py` 新增赋值操作示例，覆盖实体表/关系表的按索引赋值、按查询条件赋值（`where_set`）、批量赋值（`assign`）与 `view` 回写语义。
