# 银行风险传染模型算法

本文档描述 `demos/bank_contagion_model.py` 中抽取出的可复用银行风险传染模型。该模型可以单独运行，也可以被教程、demo、实验脚本或多后端压力测试复用。多后端压力测试通过 `demos/bank_contagion_workload.py` 适配该模型，而不是直接把经济模型算法写入压测编排器。

## 关系方向

模型使用两张 RECS `Relation` 边表：

| 关系 | src | dst | amount |
|:---|:---|:---|:---|
| `bank_firm_rel` | 银行 | 厂商/贷款资产 | 银行对该资产的敞口 |
| `bank_bank_rel` | 债权银行 | 债务银行 | 债权银行借给债务银行的金额 |

银行间边方向固定为：

```text
src = creditor / lender
dst = debtor / borrower
```

因此，当债务银行 `j` 破产时，损失沿着所有 `i -> j` 的边传给债权银行 `i`。

## 资产负债表

每家银行 `i` 的简化资产负债表为：

```text
A_exb_i = sum_a H_ia
A_IB_i  = sum_j E_ij
Z_IB_i  = sum_j E_ji
capital_i = A_exb_i + A_IB_i - Z_IB_i - Z_deposit_i
```

其中：

- `H_ia` 是银行 `i` 持有厂商/贷款资产 `a` 的敞口。
- `E_ij` 是银行 `i` 对银行 `j` 的银间债权。
- `Z_deposit_i` 是外部存款负债。

模型初始化时会从两张关系表重新汇总 `A_exb`、`A_IB`、`Z_IB`，并按目标资本率生成 `Z_deposit`，使初始状态具有正资本缓冲。

## 外部冲击：共同资产渠道

每个宏观步随机选择一组厂商/贷款资产作为外部冲击对象：

```text
shock_a ~ Bernoulli(p)
lambda_a ~ U(lambda_low, lambda_high)
```

被冲击资产的价值按比例减值：

```text
H_ia' = H_ia * (1 - lambda_a)
loss_i^asset = sum_a H_ia * lambda_a
```

如果多家银行共同持有同一资产，它们会同时遭受损失。这是共同资产持有或共同借款人违约渠道。

## 破产判定

每轮重估资本：

```text
capital_i = A_exb_i + A_IB_i - Z_IB_i - Z_deposit_i
```

若：

```text
capital_i < 0
```

则银行 `i` 被标记为 `bankrupt`，并从活跃实体集合中 `disable`。破产银行进入吸收态，后续不再作为冲击目标，也不再重复传播。

## 银行间清算传播

对本轮新增破产的债务银行 `j`，先计算资不抵债缺口：

```text
gap_j = max(-capital_j, 0)
```

默认清算规则为 `interbank_first`，即缺口先冲击银行间债权人：

```text
D_IB_j = min(gap_j, Z_IB_j)
LGD_j = D_IB_j / Z_IB_j
loss_ij = E_ij * LGD_j
```

其中 `loss_ij` 是债权银行 `i` 因债务银行 `j` 破产承受的银间资产损失。

可选清算规则 `pari_passu` 会让银行间债权人与其他债权按负债比例分担缺口：

```text
D_IB_j = min(gap_j * Z_IB_j / (Z_IB_j + Z_deposit_j), Z_IB_j)
```

## 破产银行火售：内生共同资产渠道

模型还提供一个简化火售机制。若本轮新增破产银行持有资产 `a`，则资产 `a` 的价格受到抛售压力：

```text
pressure_a = sum_{j in defaulted} H_ja / sum_i H_ia
price_drop_a = 1 - exp(-kappa * pressure_a)
```

其他仍存续银行因共同持有该资产遭受额外损失：

```text
loss_i^firesale = sum_a H_ia * price_drop_a
```

当 `fire_sale_kappa = 0` 时，该渠道关闭。

## 轮次逻辑

每个宏观步执行：

```text
1. 随机冲击厂商/贷款资产
2. 经 bank_firm_rel 汇总银行资产损失
3. 找到新增破产银行
4. 新增破产银行通过 bank_bank_rel 向债权银行传播损失
5. 新增破产银行引发共同资产火售损失
6. 重复 3-5，直到无新增破产银行或达到 max_cascade_rounds
```

关键状态集合：

```text
bankrupt_i   = 银行是否已经破产清算
propagated_i = 银行是否已经传播过破产损失
active_i     = not bankrupt_i
```

当前实现中，新增破产银行在破产轮次传播一次，随后不再重复传播。

## 蒙特卡洛随机计划

为了做严谨的多后端比较，模型支持将随机性前置为显式输入：

| 数据结构 | 含义 |
|:---|:---|
| `BankContagionScenarioData` | 后端无关的场景随机数组，包括网络结构、敞口金额、银行风险权重、厂商属性、初始资本率 |
| `BankContagionShockPlan` | 一条蒙特卡洛路径的预生成冲击数组，包括每轮资产损失率和 reset 资本率 |

推荐的多后端测试流程是：

```text
1. 使用统一 seed 生成一份 BankContagionScenarioData
2. 使用统一 seed 生成 K 条 BankContagionShockPlan
3. 对每个 backend：
   3.1 从同一份 ScenarioData 构造该 backend 的 RECS 实体池和关系表
   3.2 对第 k 条 MC 路径，使用同一个 ShockPlan[k]
```

这样可以保证：

- 同一条蒙特卡洛路径在 NumPy、PyTorch、MLX、JAX 等后端上的随机值完全一致。
- 不同蒙特卡洛路径之间的随机值不同。
- 性能差异主要来自后端计算和数据结构操作，而不是随机抽样差异。

## 复用接口

单独运行：

```bash
python3 demos/bank_contagion_model.py \
  --n-banks 5000 --n-firms 2000 \
  --density 0.001 --steps 10 \
  --backend numpy
```

在其他 demo 中复用：

```python
from demos.bank_contagion_model import (
    BankContagionModel,
    BankContagionParams,
    build_model_from_scenario_data,
    create_contagion_scenario,
    generate_monte_carlo_plans,
    generate_scenario_data,
)

params = BankContagionParams(max_cascade_rounds=8, fire_sale_kappa=0.25)
scenario = generate_scenario_data(
    n_banks=5000,
    n_firms=2000,
    density=0.001,
    seed=42,
    params=params,
)
plans = generate_monte_carlo_plans(
    n_runs=10,
    n_steps=20,
    n_banks=5000,
    n_firms=2000,
    seed=43,
    params=params,
)
model = build_model_from_scenario_data(
    scenario,
    backend="numpy",
    params=params,
    shock_plan=plans[0],
)
metrics = model.run(10)
```

如果已有实体池和关系表：

```python
model = BankContagionModel(
    bank_pool=bank_pool,
    firm_pool=firm_pool,
    bank_bank_rel=bank_bank_rel,
    bank_firm_rel=bank_firm_rel,
    params=params,
)
metrics = model.run(n_steps)
```
