# 多后端基准测试：银行网络风险传染模型

## 概述

`demos/demo_multi_backend.py` 是一个面向银行间风险传染场景的多后端计算基准测试，用于验证 RECS 框架在不同计算后端（NumPy、PyTorch、MLX、JAX）上的性能特征与功能正确性。

该 demo 以**银行关联网络风险传染模型**（Bank Interbank Risk Contagion Model）为背景，构建了一个包含银行实体池、厂商实体池、银间拆借关系和银行-厂商信贷关系的双层金融网络，并在此网络上执行多轮风险传播模拟。

## 模型背景

### 经济直觉

现代金融体系中，银行通过同业拆借市场形成复杂的债权债务网络。当一家银行遭受外部冲击（如贷款违约、市场暴跌）时，其资产减值可能导致资本不足乃至违约。而一家银行的违约会通过银间拆借网络传导至其债权人银行，引发连锁反应。这种级联违约机制是系统性金融风险的核心研究对象。

### 模型假设

1. 银行之间通过同业拆借形成有向加权网络（银间资产/负债）
2. 银行对厂商部门持有贷款类资产
3. 银行资本 = 总资产 - 总负债
4. 资本为负的银行被判定为违约，并被禁用
5. 违约银行的银间负债部分损失由债权人承担（对手方风险）
6. 外部冲击表现为银行资产的随机减值

## 数学模型

### 银行资产负债表

每家银行 $i$ 的简化资产负债表：

| 资产 (Assets) | 负债 (Liabilities) |
|:---|:---|
| 对厂商贷款 $A_{\text{exb},i}$ | 存款 $Z_{\text{deposit},i}$ |
| 银间拆出 $A_{\text{IB},i}$ | 银间拆入 $Z_{\text{IB},i}$ |
| | 资本 $C_i$ |
| | 权益 $E_i$ |

恒等式：

$$A_{\text{exb},i} + A_{\text{IB},i} = Z_{\text{deposit},i} + Z_{\text{IB},i} + C_i + E_i$$

### 资本充足条件

银行 $i$ 的风险调整后资本充足率：

$$\text{CAR}_i = \frac{C_i}{w_i \cdot A_{\text{exb},i}}$$

其中 $w_i$ 为风险权重。当 $\text{CAR}_i < \theta$（阈值）时，银行被视为资本不足。

### 外部冲击

在每轮模拟中，随机选择 5% 的银行施加外部冲击：

$$A_{\text{exb},i}^{(t+1)} = A_{\text{exb},i}^{(t)} \times (1 - \lambda_i)$$

其中 $\lambda_i \sim \mathcal{U}(0.2, 0.5)$ 为随机损失率。

### 对手方风险传导

净负债银行 $i$（$Z_{\text{IB},i} > A_{\text{IB},i}$）按比例对其银间资产减值：

$$A_{\text{IB},i}^{(t+1)} = A_{\text{IB},i}^{(t)} \times (1 - \delta_i)$$

其中 $\delta_i \sim \mathcal{U}(0.1, 0.3)$ 为违约损失率。

### 违约判定

银行资本重估：

$$C_i^{(t+1)} = (A_{\text{exb},i}^{(t+1)} + A_{\text{IB},i}^{(t+1)}) - (Z_{\text{IB},i}^{(t)} + Z_{\text{deposit},i}^{(t)})$$

当 $C_i^{(t+1)} < 0$ 时，银行被标记为违约并从活跃池中禁用。

## 算法伪代码

```
算法: 银行网络风险传染模拟
────────────────────────────────────────────────────
输入: 银行数 N, 厂商数 M, 关系密度 d, 迭代轮数 T

 1. 初始化银行实体池 (N × 7属性)
 2. 初始化厂商实体池 (M × 3属性)
 3. 生成银间拆借关系 (N×N 稀疏矩阵, 密度 d)
 4. 生成银行-厂商信贷关系 (N×M 稀疏矩阵, 密度 d)
 5. 
 6. for t = 1 to T:
 7.     // ── 阶段 C1: 外部冲击 ──
 8.     S ← 随机选取 ⌈N×5%⌉ 家银行
 9.     for each i ∈ S:
10.         λ_i ← 从 U(0.2, 0.5) 采样
11.         A_exb[i] ← A_exb[i] × (1 - λ_i)
12.     
13.     // ── 阶段 C2: 银间传导 ──
14.     D ← {i | Z_IB[i] > A_IB[i]}    // 净负债银行
15.     for each i ∈ D:
16.         δ_i ← 从 U(0.1, 0.3) 采样
17.         A_IB[i] ← A_IB[i] × (1 - δ_i)
18.     
19.     // 资本重估
20.     for i = 1 to N:
21.         Assets  ← A_exb[i] + A_IB[i]
22.         Liab    ← Z_IB[i] + Z_deposit[i]
23.         capital[i] ← Assets - Liab
24.     
25.     // ── 阶段 C3: 违约判定 ──
26.     defaulted ← {i | capital[i] < 0}
27.     disable(defaulted)
28.     
29.     // 周期性重置
30.     if t mod 10 == 0:
31.         enable(all)
32.         reset(capital, equity)
33. 
34. 输出: 完成 T 轮模拟的性能指标 (ops/s)
```

## 基准测试架构

### 三大测试组

| 组别 | 名称 | 覆盖操作 | 计数量纲 |
|:---|:---|:---|:---|
| **A** | 实体池 | 批量添加、条件查询、where_set、enable/disable | K ops/s |
| **B** | 关系表 | 稠密→稀疏转换、批量加边、uid查询、行列聚合、排序Top-K、边删除 | K ~ M ops/s |
| **C** | 风险传染 | 外部冲击→银间传导→违约判定的完整模拟链路 | K ops/s |

### 测试流程

```
for each backend ∈ {numpy, torch, mlx, jax}:
    1. 创建 bank_pool (N 银行, 7 属性)
    2. 执行 A 组基准 (N 次实体池操作)
    3. 创建 bank_bank_rel (银间拆借, 稀疏密度 d)
    4. 创建 firm_pool (M 厂商, 3 属性)
    5. 创建 bank_firm_rel (银行-厂商信贷, 稀疏密度 d)
    6. 执行 B 组基准 (N 次关系表操作)
    7. 执行 C 组基准 (N 轮风险传染模拟)
展示跨后端汇总对比表
```

### 计数量纲说明

- **ops = operations**：每次 get_attr / assign / query 等 API 调用计为一次操作
- **ops/s** = 总操作数 ÷ 总耗时（秒）
- 测试规模越大，ops/s 越能反映后端真实吞吐量

指标格式：
- `> 1e6 ops/s` → 以 `M ops/s` 显示
- `< 1e6 ops/s` → 以 `K ops/s` 显示

## 后端特性说明

### 各后端在 macOS M 芯片上的表现

| 后端 | 设备 | 特点 | 写操作 | 读操作 |
|:---|:---|:---|:---|:---|
| **NumPy** | CPU | 基础后端，始终可用 | 快速（原地修改） | 快速 |
| **MLX** | MPS (Metal) | Apple 官方 Apple Silicon 框架 | 快速（原地修改） | 最快 |
| **PyTorch** | MPS (Metal) | 广泛使用的深度学习框架 | 较快（MPS 加速） | 快 |
| **JAX** | CPU | 函数式、不可变数组 | 最慢（`.at[].set()` 开销） | 快 |

### 性能预期

经验法则（Apple M 芯片）：

```
批量添加:      NumPy ≈ MLX > Torch > JAX
条件查询:      NumPy > MLX > Torch > JAX
条件赋值:      NumPy ≈ MLX > Torch >> JAX
排序 Top-K:    MLX >> NumPy ≈ Torch > JAX
边删除:        MLX > NumPy > Torch >> JAX
```

JAX 在写密集场景下明显落后，因为其数组不可变，每次赋值都产生新数组拷贝。

## 用法

### 基本用法

```bash
# 默认参数（5 万银行 × 5 万厂商 × 0.1% 密度 × 50 次迭代）
python demos/demo_multi_backend.py
```

### 自定义参数

```bash
# 快速验证（小型测试）
python demos/demo_multi_backend.py \
    --n-banks 5000 --n-firms 2000 \
    --density 0.01 --iterations 5

# 大规模测试
python demos/demo_multi_backend.py \
    --n-banks 100000 --n-firms 50000 \
    --density 0.0005 --iterations 30

# 只测特定后端
python demos/demo_multi_backend.py \
    --backend numpy --backend mlx

# 跳过某些测试组
python demos/demo_multi_backend.py \
    --skip-group C

# 跳过 API 演示，只跑性能基准
python demos/demo_multi_backend.py \
    --skip-group A --skip-group C
```

### 参数参考

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `--n-banks` | | int | 50000 | 银行实体数量 |
| `--n-firms` | | int | 50000 | 厂商实体数量 |
| `--density` | | float | 0.001 | 关系矩阵稀疏度（千分之一） |
| `--iterations` | `-i` | int | 50 | 每组测试的迭代次数 |
| `--backend` | `-b` | str[] | 全部可用 | 指定后端（可多次使用） |
| `--skip-group` | | str[] | — | 跳过测试组 A/B/C |

## 输出解读

### 示例输出

```
RECS 多后端计算基准测试
======================================================================
  银行实体: 5,000
  厂商实体: 2,000
  ...

──────────────────────────────────────────────────────────────────────
  后端: mlx
──────────────────────────────────────────────────────────────────────

======================================================================
  测试组 A — 实体池 (5,000 银行 × 5 次)
======================================================================
  mlx              0.019s  267651K ops/s  |  批量添加实体
  mlx              0.004s   2679K ops/s   |  条件查询 (index)
  mlx              0.003s   3367K ops/s   |  条件批量赋值 (where_set)
  mlx              0.001s   4216K ops/s   |  实体启用/禁用
```

输出字段含义：

| 字段 | 含义 |
|:---|:---|
| `mlx` | 后端名称 |
| `0.019s` | 该测试项的**总耗时** |
| `267651K ops/s` | **吞吐量**（每秒操作数） |
| `批量添加实体` | 测试项描述 |

## 文件结构

```
demos/
├── demo1.py                    # ECS 基础操作演示（位置/健康度）
├── demo2.py                    # REL + ECS 完整操作用例
├── demo3.py                    # REL 操作化简版
└── demo_multi_backend.py       # 多后端性能基准测试（本文档）

docs/
├── DEMO_MULTI_BACKEND.md       # ← 本文档
├── USER_MANUAL.md              # 用户手册
├── API_REFERENCE.md            # API 参考
├── DEVELOPER_GUIDE.md          # 开发者指南
├── CHANGELOG.md                # 变更日志
├── PRD.md                      # 产品需求文档
└── ENTITY_TABLE_QUERY_BENCHMARK.md  # 实体表查询基准
```

## 参考文献

1. Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science*, 47(2), 236-249.
2. Gai, P., & Kapadia, S. (2010). Contagion in financial networks. *Proceedings of the Royal Society A*, 466(2120), 2401-2423.
3. Upper, C. (2011). Simulation methods to assess the danger of contagion in interbank markets. *Journal of Financial Stability*, 7(3), 111-125.
