# 多后端基准测试

## 概述

`demos/demo_multi_backend.py` 是 RECS 的通用多后端压力测试入口，用于比较 NumPy、PyTorch、MLX、JAX 等计算后端在实体池、关系表和模型负载上的性能特征。

该脚本本身不定义具体领域模型。领域样例通过 workload 适配器接入：

- 压测编排器：`demos/demo_multi_backend.py`
- 默认 workload：`demos/bank_contagion_workload.py`
- 默认 workload 使用的经济模型：`demos/bank_contagion_model.py`
- 经济模型说明：[银行风险传染模型算法](BANK_RISK_CONTAGION_MODEL.md)

因此，多后端压测可以换用别的 workload；银行风险传染模型也可以被其他 demo、教程或实验程序复用。

## 测试组

| 组别 | 名称 | 责任 | 典型操作 |
|:---|:---|:---|:---|
| **A** | 实体池 | 测试实体表批量操作 | 批量添加、条件查询、where_set、enable/disable |
| **B** | 关系表 | 测试关系表批量操作 | 稠密转稀疏、批量加边、uid 查询、行列聚合、排序、删除 |
| **C** | 模型负载 | 测试 workload 提供的完整模型计算链路 | 由 workload 定义 |

当前默认 workload 是 `bank-contagion`，其 C 组负载是共同资产冲击、银行间清算和火售传播。该经济学算法不在本文档展开，见 [BANK_RISK_CONTAGION_MODEL.md](BANK_RISK_CONTAGION_MODEL.md)。

## Workload 边界

`demo_multi_backend.py` 只依赖 workload 的统一接口：

```text
config_lines()
create_entity_pool(backend)
bench_entity_pool(backend)
create_relations(backend, pool)
bench_relation(backend, pool)
bench_model(pool, relation_a, relation_b, aux_pool)
```

新增 workload 时，应把领域模型、场景构造、随机数预生成和模型运行逻辑放在独立文件中，再在 `WORKLOADS` 注册表中登记。压测主程序只负责编排后端循环、跳过测试组、计时和汇总输出。

## 随机数公平性

多后端比较不能让各后端各自抽随机数。默认 workload 遵循以下规则：

1. 在进入后端循环前，预生成一份后端无关的场景数组。
2. 在进入后端循环前，预生成多条蒙特卡洛路径的随机冲击数组。
3. 每个后端从同一份场景数组构造自己的 RECS 实体池和关系表。
4. 对同一条蒙特卡洛路径，所有后端使用完全相同的冲击数组。
5. 不同蒙特卡洛路径使用不同的预生成随机数组。

这保证性能差异主要来自后端实现和数据结构操作，而不是随机抽样差异。

## 测试流程

```text
1. 解析 CLI 参数
2. 创建 workload
3. workload 预生成共享场景和蒙特卡洛随机数组
4. for each backend:
   4.1 创建后端专属实体池
   4.2 执行 A 组实体池测试
   4.3 创建后端专属关系表
   4.4 执行 B 组关系表测试
   4.5 使用同一组蒙特卡洛随机数组执行 C 组模型负载
5. 输出跨后端汇总表
```

## 计数量纲

- **ops**：压测脚本定义的操作计数单位。
- **ops/s**：`ops / elapsed_seconds`。
- A/B 组中，ops 通常对应 API 调用次数或数据规模。
- C 组中，ops 对应 `mc_runs × iterations`，即蒙特卡洛路径数乘以每条路径的宏观轮数。

指标格式：

- `> 1e6 ops/s`：以 `M ops/s` 显示
- `< 1e6 ops/s`：以 `K ops/s` 显示

## 用法

### 基本用法

```bash
python3 demos/demo_multi_backend.py
```

### 指定后端

```bash
python3 demos/demo_multi_backend.py \
  --backend numpy --backend mlx
```

### 蒙特卡洛路径

```bash
python3 demos/demo_multi_backend.py \
  --iterations 20 \
  --mc-runs 10 \
  --seed 42
```

含义：

- `--iterations 20`：每条蒙特卡洛路径运行 20 个宏观轮次。
- `--mc-runs 10`：生成 10 条蒙特卡洛路径。
- `--seed 42`：用同一个主种子预生成场景和随机冲击数组。

### 跳过测试组

```bash
python3 demos/demo_multi_backend.py \
  --skip-group A --skip-group C
```

### 默认银行传染 workload 参数

```bash
python3 demos/demo_multi_backend.py \
  --workload bank-contagion \
  --n-banks 5000 \
  --n-firms 2000 \
  --density 0.01 \
  --iterations 5 \
  --mc-runs 3 \
  --seed 42
```

## 参数参考

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `--workload` | | str | `bank-contagion` | 压测负载名称 |
| `--n-banks` | | int | 50000 | 默认银行传染 workload 的银行实体数量 |
| `--n-firms` | | int | 50000 | 默认银行传染 workload 的厂商实体数量 |
| `--density` | | float | 0.001 | 默认银行传染 workload 的关系稀疏度 |
| `--iterations` | `-i` | int | 50 | 每组测试迭代次数；C 组表示每条 MC 路径的宏观轮数 |
| `--mc-runs` | | int | 1 | C 组蒙特卡洛路径数量 |
| `--seed` | | int | 20260711 | 预生成场景和 MC 随机数组的主种子 |
| `--backend` | `-b` | str[] | 全部可用 | 指定后端，可多次使用 |
| `--skip-group` | | str[] | 无 | 跳过测试组 A/B/C，可多次使用 |

## 文件结构

```text
demos/
├── demo_multi_backend.py          # 通用多后端压测编排器
├── bank_contagion_workload.py     # 默认银行传染 workload 适配层
└── bank_contagion_model.py        # 可复用银行风险传染模型

docs/
├── DEMO_MULTI_BACKEND.md          # 本文档：压测编排与 workload 机制
└── BANK_RISK_CONTAGION_MODEL.md   # 银行风险传染模型算法
```
