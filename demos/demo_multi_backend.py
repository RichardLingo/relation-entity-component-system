"""
@File   : demo_multi_backend.py
@Desc   : RECS generic multi-backend benchmark runner.

The runner compares RECS operations across NumPy, PyTorch, MLX, and JAX.
Domain-specific setup is delegated to a workload adapter. The default workload
is the bank contagion model, but the benchmark orchestration is intentionally
kept independent from that economic model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from bank_contagion_workload import BankContagionBenchmarkWorkload
except ModuleNotFoundError:
    from demos.bank_contagion_workload import BankContagionBenchmarkWorkload


WORKLOADS = {
    BankContagionBenchmarkWorkload.name: BankContagionBenchmarkWorkload,
}


def check_backend_available(backend_name):
    """Return whether a backend import is available."""
    try:
        if backend_name == "torch":
            import torch  # noqa: F401
            return True
        if backend_name == "mlx":
            import mlx.core as mx  # noqa: F401
            return True
        if backend_name == "jax":
            import jax.numpy as jnp  # noqa: F401
            return True
        if backend_name == "tensorflow":
            import tensorflow as tf  # noqa: F401
            return True
        if backend_name == "numpy":
            return True
    except ImportError:
        return False
    return False


def _get_available_backends(requested):
    """Return requested available backends, or all available standard backends."""
    if requested:
        return [b for b in requested if check_backend_available(b)]
    all_bk = ["numpy", "torch", "mlx", "jax"]
    return [b for b in all_bk if check_backend_available(b)]


def _enable_jax_x64():
    """Enable JAX x64 so JAX follows the same dtype assumptions as other backends."""
    if check_backend_available("jax"):
        import jax
        jax.config.update("jax_enable_x64", True)


def _format_ops(ops, unit="M"):
    if unit == "M":
        return f"{ops:.1f}M"
    if unit == "K":
        return f"{ops:.0f}K"
    return f"{ops:.0f}"


def _print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_result(backend, elapsed, label, n_ops=None):
    if n_ops is not None:
        ops = n_ops / elapsed
        if ops > 1e6:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'M')} ops/s  |  {label}")
        else:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'K')} ops/s  |  {label}")
    else:
        print(f"  {backend:12s}  {elapsed:8.3f}s  |  {label}")


def run_benchmark(backend, workload, skip_groups):
    """Run all enabled benchmark groups for one backend."""
    print(f"\n{'─' * 70}")
    print(f"  后端: {backend}")
    print(f"{'─' * 70}")

    all_results = {}

    if "A" not in skip_groups:
        _print_header(f"测试组 A — 实体池 ({workload.iterations} 次)")
        pool, res_a = workload.bench_entity_pool(backend)
        all_results.update(res_a)
        for _, (label, elapsed, n_ops) in res_a.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        pool = workload.create_entity_pool(backend)

    if "B" not in skip_groups:
        _print_header(f"测试组 B — 关系表 ({workload.iterations} 次)")
        bb_rel, bf_rel, firm_pool, res_b = workload.bench_relation(backend, pool)
        all_results.update(res_b)
        for _, (label, elapsed, n_ops) in res_b.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        bb_rel, bf_rel, firm_pool = workload.create_relations(backend, pool)

    if "C" not in skip_groups:
        _print_header(f"测试组 C — 模型负载 ({workload.mc_runs} 条 MC × {workload.iterations} 轮)")
        res_c = workload.bench_model(pool, bb_rel, bf_rel, firm_pool)
        all_results.update(res_c)
        for _, (label, elapsed, n_ops) in res_c.items():
            _print_result(backend, elapsed, label, n_ops)

    return all_results


def print_summary(all_backend_results, skip_groups):
    """Print cross-backend summary."""
    print(f"\n\n{'=' * 70}")
    print("  跨后端性能汇总")
    print(f"{'=' * 70}")

    test_items = set()
    for results in all_backend_results.values():
        test_items.update(results.keys())

    group_order = {"A": [], "B": [], "C": []}
    for item in sorted(test_items):
        group = item[0]
        if group in group_order:
            group_order[group].append(item)

    for group, items in group_order.items():
        if group in skip_groups:
            continue
        group_names = {"A": "实体池", "B": "关系表", "C": "模型负载"}
        print(f"\n  ┌─ 测试组 {group} — {group_names[group]}")
        for item in items:
            first_backend = list(all_backend_results.keys())[0]
            label = all_backend_results[first_backend][item][0]
            print(f"  │  {item}  {label}")
            print(f"  │  {'─' * 50}")
            for bk, results in all_backend_results.items():
                if item in results:
                    _, elapsed, n_ops = results[item]
                    if n_ops > 0:
                        ops = n_ops / elapsed
                        if ops > 1e6:
                            print(f"  │    {bk:12s}  {elapsed:8.3f}s  {_format_ops(ops, 'M'):>8s} ops/s")
                        else:
                            print(f"  │    {bk:12s}  {elapsed:8.3f}s  {_format_ops(ops, 'K'):>8s} ops/s")
            print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RECS 多后端计算基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demos/demo_multi_backend.py
  python demos/demo_multi_backend.py --n-banks 100000 --n-firms 50000
  python demos/demo_multi_backend.py --density 0.0005 --iterations 30 --mc-runs 5
  python demos/demo_multi_backend.py --backend mlx
  python demos/demo_multi_backend.py --skip-group A --skip-group C
        """,
    )
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="bank-contagion",
                        help="压测负载名称（默认: bank-contagion）")
    parser.add_argument("--n-banks", type=int, default=50000,
                        help="默认 bank-contagion 负载的银行实体数量（默认: 50000）")
    parser.add_argument("--n-firms", type=int, default=50000,
                        help="默认 bank-contagion 负载的厂商实体数量（默认: 50000）")
    parser.add_argument("--density", type=float, default=0.001,
                        help="关系矩阵稀疏度（默认: 0.001，即千分之一）")
    parser.add_argument("--iterations", "-i", type=int, default=50,
                        help="每组测试的迭代次数；C 组中表示每条蒙特卡洛路径的宏观轮数（默认: 50）")
    parser.add_argument("--mc-runs", type=int, default=1,
                        help="C 组蒙特卡洛路径数量；同一路径在所有后端复用同一份预生成随机数组（默认: 1）")
    parser.add_argument("--seed", type=int, default=20260711,
                        help="预生成场景和蒙特卡洛随机数组的主种子（默认: 20260711）")
    parser.add_argument("--backend", "-b", action="append", dest="backends",
                        help="指定后端（可多次使用，默认: 全部可用后端）")
    parser.add_argument("--skip-group", action="append", dest="skip_groups",
                        default=[], choices=["A", "B", "C"],
                        help="跳过指定测试组（可多次使用）")
    return parser.parse_args()


def main():
    args = parse_args()
    backends = _get_available_backends(args.backends)
    skip_groups = set(args.skip_groups)
    workload_cls = WORKLOADS[args.workload]
    workload = workload_cls(
        n_banks=args.n_banks,
        n_firms=args.n_firms,
        density=args.density,
        iterations=args.iterations,
        mc_runs=args.mc_runs,
        seed=args.seed,
    )

    print("RECS 多后端计算基准测试")
    print("=" * 70)
    for label, value in workload.config_lines():
        print(f"  {label}: {value}")
    print(f"  测试后端: {', '.join(backends)}")
    active_groups = [g for g in ["A", "B", "C"] if g not in skip_groups]
    print(f"  测试组:   {', '.join(active_groups)}")
    print(f"  跳过组:   {', '.join(skip_groups) if skip_groups else '无'}")
    print()

    all_results = {}
    for bk in backends:
        _enable_jax_x64()
        all_results[bk] = run_benchmark(bk, workload, skip_groups)

    print_summary(all_results, skip_groups)

    print(f"\n{'=' * 70}")
    print("  所有基准测试完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
