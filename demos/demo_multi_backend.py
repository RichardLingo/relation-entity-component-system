"""
@File   : demo_multi_backend.py
@Desc   : RECS 多后端计算基准测试

展示 RECS 框架在不同计算后端（NumPy、PyTorch、MLX、JAX）上的完整性能特征。
覆盖以下场景（以银行关联网络风险传染模型为背景）：

测试组 A — 实体池（Entity Pool）大规模操作
  A1: 批量添加实体（多属性：银行资产/负债/资本）
  A2: 条件查询与索引（index / query / where_set）
  A3: 多属性批量更新（assign）
  A4: 实体启用/禁用（enable / disable）

测试组 B — 关系表（Relation）大规模操作
  B1: 稠密矩阵 → 稀疏关系表转换（dense_matrix_to_relation）
  B2: 批量添加边（多属性：金额、期限、利率）
  B3: 按 src/dst uid 查询与筛选（index / query / take）
  B4: 行和/列和聚合（row_sum / col_sum / row_count）
  B5: 排序与 Top-K（argsort_by / take）
  B6: 边删除（remove_by_indices）

测试组 C — 银行网络风险传染模拟
  C1: 外部冲击 → 银行资产减值 → 银间风险传导
  C2: 多轮迭代：资产重估 → 违约判定 → 级联传播

用法：
    python demos/demo_multi_backend.py                                    # 默认：5 万银行 × 5 万厂商 × 50 次
    python demos/demo_multi_backend.py --n-banks 100000                   # 10 万银行
    python demos/demo_multi_backend.py --n-firms 50000                    # 5 万厂商
    python demos/demo_multi_backend.py --density 0.001                    # 关系稀疏度 0.1%
    python demos/demo_multi_backend.py --iterations 100                   # 100 次模拟迭代
    python demos/demo_multi_backend.py --skip-group A --skip-group C      # 只跑 B 组
    python demos/demo_multi_backend.py --backend mlx                      # 只测 MLX
    python demos/demo_multi_backend.py --backend numpy --backend torch    # 只测 NumPy + Torch
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

# 确保可以导入 recs 模块
this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from recs import RECS, Relation


# ============================================================
# 工具函数
# ============================================================

def check_backend_available(backend_name):
    """检查后端是否可用"""
    try:
        if backend_name == 'torch':
            import torch
            return True
        elif backend_name == 'mlx':
            import mlx.core as mx
            return True
        elif backend_name == 'jax':
            import jax.numpy as jnp
            return True
        elif backend_name == 'tensorflow':
            import tensorflow as tf
            return True
        elif backend_name == 'numpy':
            return True
    except ImportError:
        return False
    return False


def _get_available_backends(requested):
    """获取可用的后端列表。"""
    if requested:
        return [b for b in requested if check_backend_available(b)]
    all_bk = ['numpy', 'torch', 'mlx', 'jax']
    return [b for b in all_bk if check_backend_available(b)]


def _enable_jax_x64():
    """启用 JAX 64 位模式。"""
    if check_backend_available('jax'):
        import jax
        jax.config.update("jax_enable_x64", True)


def _format_ops(ops, unit='M'):
    """格式化 ops/s 数值。"""
    if unit == 'M':
        return f"{ops:.1f}M"
    elif unit == 'K':
        return f"{ops:.0f}K"
    return f"{ops:.0f}"


def _print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_result(backend, elapsed, label, n_ops=None):
    """统一输出性能结果。"""
    if n_ops is not None:
        ops = n_ops / elapsed
        if ops > 1e6:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'M')} ops/s  |  {label}")
        else:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'K')} ops/s  |  {label}")
    else:
        print(f"  {backend:12s}  {elapsed:8.3f}s  |  {label}")


# ============================================================
# 测试组 A — 实体池大规模操作
# ============================================================

def bench_entity_pool(backend, n_banks, n_firms, n_iterations):
    """实体池基准测试：多属性银行实体池的大规模操作。"""
    results = {}

    # ---- A1: 批量添加实体 ----
    _enable_jax_x64()
    attr_types = {
        'name': 'U20',
        'A_exb': np.float32,      # 对厂商的资产
        'A_IB': np.float32,       # 银间资产
        'Z_IB': np.float32,       # 银间负债
        'Z_deposit': np.float32,  # 存款
        'capital': np.float32,    # 资本
        'equity': np.float32,     # 所有者权益
        'risk_weight': np.float32,# 风险权重
    }

    t0 = time.time()
    pool = RECS(capacity=n_banks, attr_dtypes=attr_types, backend=backend)
    pool.add(n_banks,
             name=[f'Bank_{i}' for i in range(n_banks)],
             A_exb=np.random.uniform(500, 5000, n_banks).astype(np.float32),
             A_IB=np.random.uniform(100, 3000, n_banks).astype(np.float32),
             Z_IB=np.random.uniform(100, 3000, n_banks).astype(np.float32),
             Z_deposit=np.random.uniform(1000, 10000, n_banks).astype(np.float32),
             capital=np.random.uniform(200, 2000, n_banks).astype(np.float32),
             equity=np.random.uniform(300, 3000, n_banks).astype(np.float32),
             risk_weight=np.random.uniform(0.3, 1.0, n_banks).astype(np.float32))
    t1 = time.time()
    results['A1_add'] = ('批量添加实体', t1 - t0, n_banks)

    # ---- A2: 条件查询与索引 ----
    t0 = time.time()
    for _ in range(n_iterations):
        # 查询资本充足率不足的银行（capital / risk_weight < threshold）
        idxs = pool.index(
            lambda p: (p.get_attr('capital') / (p.get_attr('risk_weight') + 1e-10)) < 500,
            return_='indices',
            include_active_only=True
        )
        # 查询高银间负债的银行
        idxs2 = pool.index(
            lambda p: p.get_attr('Z_IB') > 1500,
            return_='indices',
            include_active_only=True
        )
    t1 = time.time()
    results['A2_query'] = ('条件查询 (index)', t1 - t0, n_iterations * 2)

    # ---- A3: 多属性批量更新（where_set） ----
    t0 = time.time()
    for _ in range(n_iterations):
        # 对银间负债高的银行增加资本缓冲
        pool.where_set(
            lambda p: p.get_attr('Z_IB') > 2000,
            {'capital': 500.0, 'equity': 800.0},
            include_active_only=True,
        )
        # 对资产低的银行注入流动性
        pool.where_set(
            lambda p: p.get_attr('A_exb') < 1000,
            {'A_exb': 1000.0, 'A_IB': 500.0},
            include_active_only=True,
        )
    t1 = time.time()
    results['A3_where_set'] = ('条件批量赋值 (where_set)', t1 - t0, n_iterations * 2)

    # ---- A4: 实体启用/禁用 ----
    t0 = time.time()
    for _ in range(n_iterations):
        # 模拟违约：禁用资本为负的银行
        idxs_default = pool.index(
            lambda p: p.get_attr('capital') < 0,
            return_='indices',
            include_active_only=True
        )
        if len(idxs_default) > 0:
            pool.disable(idxs_default)
        # 模拟重组：重新启用部分银行
        if len(idxs_default) > 0:
            pool.enable(idxs_default[:max(1, len(idxs_default)//2)])
    t1 = time.time()
    results['A4_enable_disable'] = ('实体启用/禁用', t1 - t0, n_iterations)

    return pool, results


# ============================================================
# 测试组 B — 关系表大规模操作
# ============================================================

def bench_relation(backend, pool, n_firms, density, n_iterations):
    """关系表基准测试：银间拆借 + 银行-厂商信贷关系。"""
    results = {}
    n_banks = pool.size

    # ---- B1: 稠密矩阵 → 稀疏关系表 ----
    t0 = time.time()
    # 银间拆借矩阵（稀疏）：随机生成有向边
    n_possible_edges = n_banks * n_banks
    n_edges_ib = max(1, int(n_possible_edges * density))
    src_idx = np.random.randint(0, n_banks, n_edges_ib)
    dst_idx = np.random.randint(0, n_banks, n_edges_ib)
    # 避免自环
    self_loop = src_idx == dst_idx
    src_idx = src_idx[~self_loop]
    dst_idx = dst_idx[~self_loop]
    n_edges_ib = len(src_idx)

    # 构建稠密矩阵（仅用于 dense_matrix_to_relation 测试）
    mat_ib = np.zeros((n_banks, n_banks), dtype=np.float32)
    mat_ib[src_idx, dst_idx] = np.random.uniform(10, 1000, n_edges_ib).astype(np.float32)

    bank_bank_rel = pool.dense_matrix_to_relation(
        mat_ib, dst_pool=pool,
        relation_name='interbank',
        attr_name='amount', threshold=0.0
    )
    t1 = time.time()
    results['B1_dense_to_rel'] = ('稠密→稀疏转换', t1 - t0, n_edges_ib)

    # ---- B2: 批量添加边（银行-厂商信贷） ----
    t0 = time.time()
    n_edges_bf = max(1, int(n_banks * n_firms * density))
    src_bf = np.random.randint(0, n_banks, n_edges_bf)
    dst_bf = np.random.randint(0, n_firms, n_edges_bf)
    amounts = np.random.uniform(50, 2000, n_edges_bf).astype(np.float32)
    maturities = np.random.randint(1, 365, n_edges_bf).astype(np.float32)
    rates = np.random.uniform(0.03, 0.12, n_edges_bf).astype(np.float32)

    # 创建厂商池
    firm_attr = {'name': 'U20', 'production': np.float32, 'DA': np.float32}
    firm_pool = RECS(capacity=n_firms, attr_dtypes=firm_attr, backend=backend)
    firm_pool.add(n_firms,
                  name=[f'Firm_{i}' for i in range(n_firms)],
                  production=np.random.uniform(0.5, 2.0, n_firms).astype(np.float32),
                  DA=np.random.uniform(100, 5000, n_firms).astype(np.float32))

    # 直接创建 Relation（避免稠密矩阵）
    bank_firm_rel = Relation(
        'credit', capacity=max(8, n_edges_bf),
        attr_dtypes={'amount': np.float32, 'maturity': np.float32, 'rate': np.float32},
        backend=backend
    )
    bank_firm_rel.add(
        src_bf.astype(np.int64), dst_bf.astype(np.int64),
        amount=amounts, maturity=maturities, rate=rates
    )
    t1 = time.time()
    results['B2_add_edges'] = ('批量添加边', t1 - t0, n_edges_bf)

    # ---- B3: 按 src/dst uid 查询与筛选 ----
    t0 = time.time()
    for _ in range(n_iterations):
        # 按源银行查询
        sample_uid = int(pool.d['i'][np.random.randint(0, n_banks)])
        idxs = bank_firm_rel.index(src_uid=sample_uid, return_='indices')
        # 按金额筛选
        idxs2 = bank_firm_rel.index(
            edge_pred=lambda r: r.get_attr('amount') > 500,
            return_='indices'
        )
        # 组合条件
        idxs3 = bank_firm_rel.index(
            src_uid=sample_uid,
            edge_pred=lambda r: r.get_attr('amount') > 500,
            return_='indices'
        )
    t1 = time.time()
    results['B3_query'] = ('按 uid/属性查询', t1 - t0, n_iterations * 3)

    # ---- B4: 行和/列和聚合 ----
    t0 = time.time()
    for _ in range(n_iterations):
        bank_uid_to_pos = bank_firm_rel.build_uid_to_pos(pool.uid_array())
        firm_uid_to_pos = bank_firm_rel.build_uid_to_pos(firm_pool.uid_array())
        row_sums = bank_firm_rel.row_sum('amount', bank_uid_to_pos, n_rows=n_banks)
        col_sums = bank_firm_rel.col_sum('amount', firm_uid_to_pos, n_cols=n_firms)
        row_deg = bank_firm_rel.row_count(bank_uid_to_pos, n_rows=n_banks)
        col_deg = bank_firm_rel.col_count(firm_uid_to_pos, n_cols=n_firms)
    t1 = time.time()
    results['B4_agg'] = ('行和/列和聚合', t1 - t0, n_iterations * 4)

    # ---- B5: 排序与 Top-K ----
    t0 = time.time()
    for _ in range(n_iterations):
        order = bank_firm_rel.argsort_by('amount', ascending=False)
        top_k = min(100, bank_firm_rel.size)
        top_edges = bank_firm_rel.take(order[:top_k])
    t1 = time.time()
    results['B5_sort'] = ('排序与 Top-K', t1 - t0, n_iterations)

    # ---- B6: 边删除 ----
    t0 = time.time()
    for _ in range(min(n_iterations, 10)):  # 删除操作较慢，减少次数
        if bank_firm_rel.size > 1:
            # 删除金额最小的边
            order_asc = bank_firm_rel.argsort_by('amount', ascending=True)
            n_remove = max(1, bank_firm_rel.size // 100)
            bank_firm_rel.remove_by_indices(order_asc[:n_remove])
    t1 = time.time()
    results['B6_remove'] = ('边删除', t1 - t0, min(n_iterations, 10))

    return bank_bank_rel, bank_firm_rel, firm_pool, results


# ============================================================
# 测试组 C — 银行网络风险传染模拟
# ============================================================

def bench_risk_contagion(backend, pool, bank_bank_rel, n_iterations):
    """风险传染模拟：外部冲击 → 资产减值 → 银间传导 → 级联违约。"""
    results = {}
    n_banks = pool.size

    # 初始化风险参数
    t0 = time.time()
    for iteration in range(n_iterations):
        # ---- C1: 外部冲击 ----
        # 模拟外部冲击：随机 5% 的银行遭受资产损失
        shock_mask = np.random.random(n_banks) < 0.05
        shock_indices = np.nonzero(shock_mask)[0]
        if len(shock_indices) > 0:
            # 资产减值 20%-50%
            loss_rates = np.random.uniform(0.2, 0.5, len(shock_indices)).astype(np.float32)
            current_A_exb = pool.get_attr('A_exb')
            # 使用 backend 的 copy 方法，兼容 PyTorch（clone）等
            new_A_exb = pool.backend.copy(current_A_exb)
            new_A_exb_np = pool.backend.to_numpy(new_A_exb).copy()
            new_A_exb_np[shock_indices] *= (1.0 - loss_rates)
            pool.assign(shock_indices, {'A_exb': new_A_exb_np[shock_indices]})

        # ---- C2: 银间风险传导 ----
        # 读取当前银间资产和负债
        current_A_IB = pool.get_attr('A_IB')
        current_Z_IB = pool.get_attr('Z_IB')
        current_capital = pool.get_attr('capital')
        current_equity = pool.get_attr('equity')

        # 转换为 numpy 以便进行条件判断
        current_A_IB_np = pool.backend.to_numpy(current_A_IB)
        current_Z_IB_np = pool.backend.to_numpy(current_Z_IB)

        # 识别净负债银行（Z_IB > A_IB）
        debtor_mask = current_Z_IB_np > current_A_IB_np
        debtor_indices = np.nonzero(debtor_mask)[0]

        if len(debtor_indices) > 0:
            # 净负债银行按比例减值其银间资产（对手方风险）
            default_ratio = np.random.uniform(0.1, 0.3, len(debtor_indices)).astype(np.float32)
            new_A_IB_np = current_A_IB_np.copy()
            new_A_IB_np[debtor_indices] *= (1.0 - default_ratio)
            pool.assign(debtor_indices, {'A_IB': new_A_IB_np[debtor_indices]})

            # 更新资本 = 资产 - 负债（简化）
            current_A_exb_np = pool.backend.to_numpy(pool.get_attr('A_exb'))
            current_A_IB_updated = pool.backend.to_numpy(pool.get_attr('A_IB'))
            current_Z_IB_np = pool.backend.to_numpy(pool.get_attr('Z_IB'))
            current_Z_deposit_np = pool.backend.to_numpy(pool.get_attr('Z_deposit'))
            
            total_assets = current_A_exb_np + current_A_IB_updated
            total_liab = current_Z_IB_np + current_Z_deposit_np
            new_capital = (total_assets - total_liab).astype(np.float32)
            pool.assign(list(range(n_banks)), {'capital': new_capital})

        # ---- C3: 违约判定与级联 ----
        # 资本为负的银行标记为违约
        defaulted = pool.index(
            lambda p: p.get_attr('capital') < 0,
            return_='indices',
            include_active_only=True
        )
        if len(defaulted) > 0:
            pool.disable(defaulted)

        # 每 10 轮重置一次（避免全部违约）
        if iteration % 10 == 0 and iteration > 0:
            pool.enable(list(range(n_banks)))
            # 重置资本
            pool.assign(list(range(n_banks)), {
                'capital': np.random.uniform(200, 2000, n_banks).astype(np.float32),
                'equity': np.random.uniform(300, 3000, n_banks).astype(np.float32),
            })

    t1 = time.time()
    results['C_contagion'] = ('风险传染模拟', t1 - t0, n_iterations)
    return results


# ============================================================
# 主基准测试调度
# ============================================================

def run_benchmark(backend, n_banks, n_firms, density, n_iterations, skip_groups):
    """对单个后端运行所有基准测试组。"""
    print(f"\n{'─' * 70}")
    print(f"  后端: {backend}")
    print(f"{'─' * 70}")

    all_results = {}

    # 测试组 A — 实体池
    if 'A' not in skip_groups:
        _print_header(f"测试组 A — 实体池 ({n_banks:,} 银行 × {n_iterations} 次)")
        pool, res_a = bench_entity_pool(backend, n_banks, n_firms, n_iterations)
        all_results.update(res_a)
        for key, (label, elapsed, n_ops) in res_a.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        # 仍然需要 pool 供后续测试
        attr_types = {
            'name': 'U20',
            'A_exb': np.float32,
            'A_IB': np.float32,
            'Z_IB': np.float32,
            'Z_deposit': np.float32,
            'capital': np.float32,
            'equity': np.float32,
            'risk_weight': np.float32,
        }
        pool = RECS(capacity=n_banks, attr_dtypes=attr_types, backend=backend)
        pool.add(n_banks,
                 name=[f'Bank_{i}' for i in range(n_banks)],
                 A_exb=np.random.uniform(500, 5000, n_banks).astype(np.float32),
                 A_IB=np.random.uniform(100, 3000, n_banks).astype(np.float32),
                 Z_IB=np.random.uniform(100, 3000, n_banks).astype(np.float32),
                 Z_deposit=np.random.uniform(1000, 10000, n_banks).astype(np.float32),
                 capital=np.random.uniform(200, 2000, n_banks).astype(np.float32),
                 equity=np.random.uniform(300, 3000, n_banks).astype(np.float32),
                 risk_weight=np.random.uniform(0.3, 1.0, n_banks).astype(np.float32))

    # 测试组 B — 关系表
    if 'B' not in skip_groups:
        _print_header(f"测试组 B — 关系表 (密度={density}, {n_iterations} 次)")
        bb_rel, bf_rel, fp, res_b = bench_relation(backend, pool, n_firms, density, n_iterations)
        all_results.update(res_b)
        for key, (label, elapsed, n_ops) in res_b.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        # 仍然需要 bank_bank_rel 供 C 组
        n_edges_ib = max(1, int(n_banks * n_banks * density))
        src_idx = np.random.randint(0, n_banks, n_edges_ib)
        dst_idx = np.random.randint(0, n_banks, n_edges_ib)
        self_loop = src_idx == dst_idx
        src_idx = src_idx[~self_loop]
        dst_idx = dst_idx[~self_loop]
        mat_ib = np.zeros((n_banks, n_banks), dtype=np.float32)
        mat_ib[src_idx, dst_idx] = np.random.uniform(10, 1000, len(src_idx)).astype(np.float32)
        bb_rel = pool.dense_matrix_to_relation(mat_ib, dst_pool=pool, relation_name='interbank', attr_name='amount', threshold=0.0)

    # 测试组 C — 风险传染模拟
    if 'C' not in skip_groups:
        _print_header(f"测试组 C — 风险传染模拟 ({n_iterations} 轮)")
        res_c = bench_risk_contagion(backend, pool, bb_rel, n_iterations)
        all_results.update(res_c)
        for key, (label, elapsed, n_ops) in res_c.items():
            _print_result(backend, elapsed, label, n_ops)

    return all_results


# ============================================================
# 汇总报告
# ============================================================

def print_summary(all_backend_results, skip_groups):
    """打印跨后端汇总对比表。"""
    print(f"\n\n{'=' * 70}")
    print("  跨后端性能汇总")
    print(f"{'=' * 70}")

    # 收集所有测试项
    test_items = set()
    for bk, results in all_backend_results.items():
        test_items.update(results.keys())

    # 按组排序
    group_order = {'A': [], 'B': [], 'C': []}
    for item in sorted(test_items):
        group = item[0]
        if group in group_order:
            group_order[group].append(item)

    for group, items in group_order.items():
        if group in skip_groups:
            continue
        group_names = {'A': '实体池', 'B': '关系表', 'C': '风险传染'}
        print(f"\n  ┌─ 测试组 {group} — {group_names[group]}")
        for item in items:
            label = all_backend_results[list(all_backend_results.keys())[0]][item][0]
            print(f"  │  {item}  {label}")
            print(f"  │  {'─' * 50}")
            for bk in all_backend_results:
                if item in all_backend_results[bk]:
                    _, elapsed, n_ops = all_backend_results[bk][item]
                    if n_ops > 0:
                        ops = n_ops / elapsed
                        if ops > 1e6:
                            print(f"  │    {bk:12s}  {elapsed:8.3f}s  {_format_ops(ops, 'M'):>8s} ops/s")
                        else:
                            print(f"  │    {bk:12s}  {elapsed:8.3f}s  {_format_ops(ops, 'K'):>8s} ops/s")
            print()


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='RECS 多后端计算基准测试 — 银行网络风险传染模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demos/demo_multi_backend.py                                           # 默认
  python demos/demo_multi_backend.py --n-banks 100000 --n-firms 50000          # 10 万银行 + 5 万厂商
  python demos/demo_multi_backend.py --density 0.0005 --iterations 30          # 更低密度
  python demos/demo_multi_backend.py --backend mlx                             # 只测 MLX
  python demos/demo_multi_backend.py --skip-group A --skip-group C             # 只跑 B 组
  python demos/demo_multi_backend.py --n-banks 5000 --n-firms 2000 --iterations 10 --skip-group C  # 快速验证
        """
    )
    parser.add_argument('--n-banks', type=int, default=50000,
                        help='银行实体数量（默认: 50000）')
    parser.add_argument('--n-firms', type=int, default=50000,
                        help='厂商实体数量（默认: 50000）')
    parser.add_argument('--density', type=float, default=0.001,
                        help='关系矩阵稀疏度（默认: 0.001，即千分之一）')
    parser.add_argument('--iterations', '-i', type=int, default=50,
                        help='每组测试的迭代次数（默认: 50）')
    parser.add_argument('--backend', '-b', action='append', dest='backends',
                        help='指定后端（可多次使用，默认: 全部可用后端）')
    parser.add_argument('--skip-group', action='append', dest='skip_groups',
                        default=[], choices=['A', 'B', 'C'],
                        help='跳过指定测试组（可多次使用）')
    return parser.parse_args()


def main():
    args = parse_args()
    backends = _get_available_backends(args.backends)
    skip_groups = set(args.skip_groups)

    print("RECS 多后端计算基准测试")
    print("=" * 70)
    print(f"  银行实体: {args.n_banks:,}")
    print(f"  厂商实体: {args.n_firms:,}")
    print(f"  关系密度: {args.density}")
    print(f"  迭代次数: {args.iterations}")
    print(f"  测试后端: {', '.join(backends)}")
    active_groups = [g for g in ['A', 'B', 'C'] if g not in skip_groups]
    print(f"  测试组:   {', '.join(active_groups)}")
    print(f"  跳过组:   {', '.join(skip_groups) if skip_groups else '无'}")
    print()

    all_results = {}
    for bk in backends:
        _enable_jax_x64()
        results = run_benchmark(bk, args.n_banks, args.n_firms,
                                args.density, args.iterations, skip_groups)
        all_results[bk] = results

    print_summary(all_results, skip_groups)

    print(f"\n{'=' * 70}")
    print("  所有基准测试完成！")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
