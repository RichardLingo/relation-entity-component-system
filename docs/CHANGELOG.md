# 变更日志

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
