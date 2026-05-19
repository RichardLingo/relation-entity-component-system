# 第一类二维数组（实体表）Query 性能评估（基准报告）

本文档评估：在类 RECS（SoA）实体表中，“**由属性反查得到索引**”（也就是 RECS Query 的核心路径）这件事的性能。

> 说明
> - 本项目的“第一类二维数组”是实体属性表：属性名 -> 1D NumPy 数组（SoA）。
> - Query 的典型形态是：`谓词（向量化比较） -> bool mask -> np.nonzero(mask)`。
> - 本报告是微基准（micro-benchmark），主要测量 **NumPy 向量化比较 + 逻辑组合 + nonzero** 的耗时量级。

---

## 1. 测试代码位置

基准测试已写入：
- `demos/demo2.py`

核心函数：
- `bench_entity_query(n_entities, rounds)`

它构造了三列：
- `A_exb: float32`（模拟连续数值属性）
- `A_IB: float32`（模拟连续数值属性）
- `o: bool`（模拟激活掩码）

查询谓词为：

```python
mask = (A_exb > 3000) & (A_IB > 1000) & o
idxs = np.nonzero(mask)[0]
```

---

## 2. 评估指标（输出字段）

`bench_entity_query` 返回：
- `n`：实体数量
- `rounds`：重复轮数
- `hit_mean`：平均命中行数（越大表示条件越宽松）
- `sec_mean/min/max`：每轮耗时（秒）
- `rows_per_sec_mean`：吞吐量（行/秒），衡量 query 扫描密集向量的速度

---

## 3. 如何解读结果（工程化结论）

### 3.1 结论 1：第一类实体表 Query 的瓶颈通常是“带宽/内存扫描”
对于 SoA 的向量化谓词，CPU 主要在连续地扫描多列数组（顺序访问）。
在这种模式下：
- 性能通常接近内存带宽上限；
- 条件再复杂，若仍能写成 NumPy 向量化表达式，性能不会“线性变差到 Python 循环级别”。

### 3.2 结论 2：mask->indices（nonzero）在命中率高时可能成为显著开销
当命中行很多时，`np.nonzero(mask)` 需要生成更大的索引数组，会带来额外内存写入与分配。
工程建议：
- 若后续只需要 mask（例如继续与别的条件组合），尽量 **延迟** 调用 `nonzero`。
- 若必须得到 indices：可以考虑秩序上减少中间 mask（例如先合并条件再 nonzero）。

### 3.3 结论 3：布尔字段的稀疏化对“存储”帮助更大，对“纯 Query 吞吐”未必更快
- 纯谓词扫描时，`bool` 数组本身已经很紧凑。
- 若布尔列极多且要频繁做集合运算（AND/OR/DIFF），位图（bitset/packbits）更有意义。

---

## 4. 如何复现实验

在 Windows PowerShell：

```powershell
$env:PYTHONPATH = "C:\Users\Ethan\CoreFiles\ProjectsFile\relation-entity-component-system"
python .\demos\demo2.py
```

运行后会看到类似：

```text
[实体表 Query 基准] {'n': ..., 'rounds': ..., 'hit_mean': ..., 'sec_mean': ..., 'rows_per_sec_mean': ...}
```

---

## 5. 后续增强建议（可选）

1. **更贴近真实 RECS Query 的形态**：
   - 增加按类别字段（int codes）的谓词
   - 增加更多列组合（3~8 列）

2. **增加对比项**：
   - `mask` 直接链式组合 vs 频繁 `nonzero`
   - `np.where(mask)` vs `np.nonzero(mask)`

3. **集成到统一 Query API**：
   - 将 `EntityPool.match(...)` 正式化，并让 Relation 查询复用同一“predicate->mask”规范。

