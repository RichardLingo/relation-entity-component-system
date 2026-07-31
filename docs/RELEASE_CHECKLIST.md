# 发布检查清单

本文件用于 RECS 发版前检查，尤其关注 API 变化、语法糖变化和用户迁移提示是否完整。

## 版本信息

- 目标版本：
- 发布日期：
- 发布负责人：
- 是否包含破坏性变更：是 / 否
- 是否包含弃用 API：是 / 否

## 文档检查

- [ ] `docs/CHANGELOG.md` 已记录本版本变更。
- [ ] 如有破坏性变更，`docs/MIGRATION_GUIDE.md` 已提供迁移步骤。
- [ ] 如有弃用 API，`docs/DEPRECATION_POLICY.md` 的流程已执行。
- [ ] `docs/API_REFERENCE.md` 已更新为当前 API。
- [ ] `docs/USER_MANUAL.md` 已更新推荐写法。
- [ ] `README.md` 的文档入口和重大升级提示已更新。
- [ ] 相关 demo 已更新，不再展示过时写法。

## 代码检查

- [ ] 旧 API 如仍保留，已发出 `DeprecationWarning`。
- [ ] warning 文案包含废弃版本、替代写法和迁移文档入口。
- [ ] 破坏性变更有测试覆盖。
- [ ] 兼容路径有测试覆盖。
- [ ] 示例脚本可运行。

## 多后端检查

- [ ] NumPy 后端通过基础测试。
- [ ] PyTorch 后端如可用，通过相关测试。
- [ ] MLX 后端如可用，通过相关测试。
- [ ] JAX 后端如可用，通过只读/支持范围内测试。
- [ ] `demos/demo_multi_backend.py` 的默认 workload 可运行。

## 发布说明模板

```markdown
## RECS vX.Y.Z

### Breaking Changes

- 无 / 列出破坏性变更。

### Migration

- 如从旧版本升级，请阅读 `docs/MIGRATION_GUIDE.md`。

### Added

- ...

### Changed

- ...

### Deprecated

- ...

### Removed

- ...

### Fixed

- ...
```

