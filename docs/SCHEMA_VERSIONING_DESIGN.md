# RECS dtype Schema 版本化设计稿（Design Draft · 只设计不实现）

> **状态**：设计稿（proposed），**本轮不实现**，仅登记设计口径供后续决策。
> **范围**：RECS 内部——为 SoA 列存引擎的 `attr_dtypes` 引入可声明、可版本化、可迁移的 **dtype schema** 契约。
> **对齐**：与 `DEPRECATION_POLICY.md` 的语义化版本与弃用流程一致；不改变现有 `RECS(capacity, attr_dtypes, backend)` 构造签名的向后兼容。

## 1. 动机与现状

**现状**：RECS 的列类型由构造入参 `attr_dtypes: dict[str, dtype]` 声明，引擎自动追加必备字段 `i`（uid, int64）与 `o`（on, bool）。dtype 信息**只在构造期存在**，运行期无统一的 schema 对象、无版本戳、无导出/校验/迁移契约。关系（`Relation`）侧同理（`src_uid`/`dst_uid` + 自定义属性列）。

**问题**：
1. 无法回答「这批列数据遵循哪个版本的 schema」——序列化/互操作时缺少自描述。
2. dtype 演进（如 `float32→float64`、新增列、字符串列宽调整）无兼容判定与迁移路径。
3. 跨后端（NumPy/PyTorch/MLX/JAX/TF）时，逻辑 dtype 与物理后端 dtype 的映射未显式化，字符串列「恒为 NumPy」的特例散落在实现里。

## 2. 设计目标 / 非目标

**目标**
- 定义**逻辑 schema**：字段名 → 逻辑 dtype（含形状/可空/语义标签可选），与物理后端解耦。
- 为 schema 附**版本戳**（`schema_version`），随引擎/关系实例可查询、可导出。
- 定义**兼容规则**（何为兼容变更、何为破坏性变更）与**迁移钩子**（旧 schema 数据 → 新 schema）。
- 保持零破坏：未显式声明 schema 的旧用法按「推断 schema + 默认版本」运行。

**非目标（本稿不做）**
- 不实现任何协议/快照序列化格式（RECS 只做数据引擎，不承载协议语义）。
- 不引入运行时/调度框架。
- 不承载任何领域/学科语义（schema 只描述数据类型，不描述业务含义）。

## 3. 核心概念

### 3.1 逻辑 dtype 与 FieldSpec

```text
FieldSpec:
  name: str                 # 列名
  logical_dtype: str        # 逻辑类型：int8..int64 / uint* / float16..64 / bool / str / object
  shape: tuple[int,...]?    # 可选：多维列的形状（标量列为空）
  nullable: bool = False    # 是否允许缺失（缺省用哨兵或掩码表达）
  semantic: str?            # 可选：语义标签（如 "uid" / "on" / "coordinate"），仅元数据、不参与计算
  encoding: str?            # 可选：序列化编码提示（如 "decimal-string"），供上层互操作参考
```

- `i` / `o` 为**引擎保留字段**，其 FieldSpec 由引擎固定生成（`i`: int64+semantic=uid；`o`: bool+semantic=on），用户不可覆盖。
- 字符串列（logical_dtype=str）恒定映射为 NumPy `U*`/`S*`（不随后端迁移）——把现有隐式特例显式写进 schema 映射表。

### 3.2 Schema 对象与版本戳

```text
Schema:
  version: str              # schema 规范版本（独立于包 semver，见 §5）
  fields: dict[str, FieldSpec]
  backend_hint: str?        # 记录导出时的物理后端（仅信息，不约束逻辑 schema）

  def fingerprint() -> str  # 字段集 + 逻辑 dtype 的稳定哈希，用于快速兼容比对
```

- `RECS` / `Relation` 实例暴露只读 `schema` 属性；`schema.version` 与 `schema.fingerprint()` 可查询。
- 构造期若传入旧式 `attr_dtypes: dict[str, dtype]`，引擎**推断**出等价 `Schema`（version=当前默认，fields 由 dtype 反推），保证向后兼容。

### 3.3 逻辑 dtype ↔ 物理后端映射表

单一映射表集中管理「逻辑 dtype → 各后端物理 dtype」，字符串列强制 NumPy。新增后端只需补该表的一列，不改引擎核心（对齐 AGENTS「后端隔离」红线）。

## 4. 兼容与迁移规则

| 变更类型 | 兼容性 | 处理 |
|---|---|---|
| 新增可选列（带 default） | 兼容（minor） | 旧数据读取时按 default 回填 |
| 列重命名 | 破坏性（major） | 需迁移钩子 `rename_map` |
| 逻辑 dtype 提升（float32→float64 / int32→int64） | 兼容（minor，无损） | 自动 upcast |
| 逻辑 dtype 收窄（float64→float32） | 破坏性（可能丢精度） | 需显式迁移声明 + 容差登记 |
| 字符串列宽变化（U5→U10） | 兼容（放宽）/ 破坏性（收窄截断） | 放宽自动；收窄需迁移声明 |
| 删除列 | 破坏性（major） | 迁移钩子 + CHANGELOG 记录 |

**迁移钩子（设计，不实现）**：`migrate(data: dict, from_version, to_version) -> dict`，按版本序应用一组声明式变换（rename / cast / fill / drop）；无声明路径的破坏性变更直接拒绝并报错（对齐 DEPRECATION_POLICY「不无提示删除」）。

## 5. 版本策略

- `schema_version` **独立于包 semver**：schema 规范版本描述「字段声明与兼容规则的形态」，包版本描述 API 面。二者解耦，避免 dtype schema 演进被迫绑定发包节奏。
- schema 版本遵循 `MAJOR.MINOR`：MAJOR=破坏性 schema 形态变更（需迁移）；MINOR=兼容性新增（新可选字段/新逻辑 dtype）。
- 每次 schema MAJOR 变更须在 `docs/CHANGELOG.md` 与 `MIGRATION_GUIDE.md` 各记一条（对齐 DEPRECATION_POLICY 流程）。

## 6. 序列化自描述（设计边界）

- Schema 可导出为纯数据（dict/JSON）作为**列数据的自描述头**，供任何上层互操作消费；但**RECS 不定义线协议/快照格式**——序列化格式与传输语义归上层，RECS 只提供 `schema.to_dict()` / `Schema.from_dict()` 的中性数据。
- 这条边界是硬红线：schema 版本化只解决「列数据类型自描述与演进」，不越界承担协议归属。

## 7. 开放问题（待后续决策）

1. `nullable` 的缺省表达：哨兵值 vs 独立掩码列 vs 后端原生 NA——影响 SoA 内存布局。
2. `semantic` 标签是否进 fingerprint（进则语义变更视为 schema 变更，可能过严）。
3. 多维列（shape 非空）的 dtype 校验粒度与 C-contiguous 约束。
4. schema 推断的歧义处理（同一 numpy dtype 可能对应多个逻辑 dtype，如 `int64` vs `uid`）。
5. 迁移钩子的声明式 DSL 形态（纯 dict 描述 vs 可调用列表）。

## 8. 落地顺序建议（若将来实现）

1. 先落 `FieldSpec` / `Schema` 只读数据类 + 由 `attr_dtypes` 推断（零破坏）。
2. 加 `schema` 属性与 `to_dict/from_dict`（自描述头）。
3. 加逻辑↔物理映射表集中化（收编现有字符串列特例）。
4. 最后加兼容判定与迁移钩子（破坏性变更 gated）。

*本稿为设计草案，不含实现；任何实现须另立任务并遵循 DEPRECATION_POLICY 与本稿 §4 兼容规则。*
