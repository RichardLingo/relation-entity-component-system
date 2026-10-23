# RECS 项目指令（canonical）

> 本文件是本项目 AI 指令的唯一正文源；CLAUDE.md、.github/copilot-instructions.md 为薄转发壳。
> 详细多根工作区导航见同目录 WORKSPACE_GUIDE.md。

## 身份与写边界（本文件适用范围）

- **本文件适用根**：`/Users/ethan/CoreFiles/ProjectsFile/relation-entity-component-system`（绝对路径，唯一）。多根工作区里其他根的指令文件不适用于本根。
- **根清单真源**：`RECS.code-workspace`（VSCode 为真源）；工作区增删根后须同步本节，勿凭记忆改。
- **本根可写**：本仓库根目录内的全部内容（源码、`docs/`、配置）；以及下列对应笔记区。
- **对应笔记区（可写）**：
  - 关联实体组件系统 → `/Users/ethan/Library/Mobile Documents/iCloud~md~obsidian/Documents/Engs_notebook/Projects/关联实体组件系统`
- **只读依赖**：（无）
- **禁止写入**：任何未在本节列出的兄弟根；禁止按文档正文出现频次推断归属。
- **归属判据**：笔记项目的工程仓库以其 `_docs-mirror/<仓库名>/` 目录名为准，远程一律取 `git config --get remote.origin.url`（UHW 唯一总设计 §9.16）。

## 项目概述

RECS（relation-entity-component-system）是一个轻量级 **SoA（Structure of Arrays）风格的 ECS 数据引擎**：按列存实体属性（存储字典 `d`：key→数组）、按表存关系（`Relation`）。主入口类 `RECS`（兼容层 `EntityPool`），支持多后端（NumPy / PyTorch / MLX / JAX / TensorFlow），所有数组计算优先向量化。包布局：`ecs/`（`engine.py` 核心）+ `recs/`（`ecsengine.py` 与 `backends/` 后端抽象），PyPI 包名 `relation-entity-component-system-kit`。

## 构建与测试

- 环境用 `uv`（见 `uv.lock`）；Python 要求 `>=3.13`，仅 NumPy 为硬依赖，其余后端为可选 extra。
- 安装可选后端：`uv pip install -e ".[torch]"`（或 `mlx` / `jax` / `tf` / `all`）；开发依赖 `uv pip install -e ".[dev]"`（含 pytest）。
- 跑测试：`pytest`（配置在 `pyproject.toml [tool.pytest.ini_options]`，`testpaths=tests`、仅收集 `test_*.py`）。多后端用例按 `list_backends()` 参数化；JAX 数组不可变，写操作测试自动跳过 jax。
- 旧脚本 `tests/smoke_test.py` / `tests/comprehensive_test.py` 为可独立运行的遗留冒烟脚本（`python tests/smoke_test.py`），**不入 pytest 收集**；新测试写 `tests/test_*.py`。

## 禁止与易踩坑

- **纯度红线**：RECS 只做**通用数据引擎**（列存、关系、批量算子、多后端），**不承载任何领域/学科语义**（不得引入游戏、金融、物理等业务词汇），**不内置协议/快照语义，不做运行时/调度框架**。
- **后端隔离**：引擎核心不得直接 `import torch/mlx/jax/tf`；一切后端差异经 `recs/backends/` 抽象，新增后端只在 `backends/` 注册。
- 必备字段 `i`（uid, int64）与 `o`（on/激活掩码, bool）由引擎管理，不得当普通业务列改写；`remove = disable`（置 `o=False`，`size` 不变）。
- 字符串列（dtype kind `U`/`S`）恒为 NumPy 数组（不随后端迁移），比较用 `array_equal` 而非 `allclose`。
- 批量优先：新增/查询/赋值走批量 API 与切片，避免逐元素 Python 循环。

## 项目结构（速查，详情见 WORKSPACE_GUIDE.md）

- 💻 本项目 · relation-entity-component-system → `./`
- `ecs/engine.py`：`RECS` / `EntityPool` / `Relation` 核心实现
- `recs/`：`ecsengine.py`（对外入口）+ `backends/`（多后端抽象与注册表）
- `tests/`：`test_smoke.py` / `test_comprehensive.py`（pytest 套件）+ 遗留 `*_test.py` 脚本
- `docs/`：API_REFERENCE / DEVELOPER_GUIDE / USER_MANUAL / MIGRATION_GUIDE / DEPRECATION_POLICY 等

