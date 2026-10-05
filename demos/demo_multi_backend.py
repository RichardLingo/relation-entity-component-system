"""
@File   : demo_multi_backend.py
@Desc   : RECS 通用多后端基准测试运行器。

该运行器比较 RECS 在 NumPy、PyTorch、MLX 和 JAX 后端上的操作性能。
领域相关的初始化交给负载适配器处理。默认负载是银行风险传染模型，
但基准测试编排刻意与该经济模型解耦。

读者可以把本文件理解为“调度层”：
- 解析命令行参数；
- 检测当前环境可用的计算后端；
- 调用负载适配器执行 A/B/C 三组测试；
- 汇总并打印每个后端的耗时和吞吐。
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
    # 后续如果增加其他业务负载，只需要在这里注册“负载名称 -> 负载类”。
    BankContagionBenchmarkWorkload.name: BankContagionBenchmarkWorkload,
}


def check_backend_available(backend_name):
    """返回指定后端是否可以导入。

    Args:
        backend_name: 后端名称，例如 ``numpy``、``torch``、``mlx``、``jax`` 或 ``tensorflow``。

    Returns:
        如果当前 Python 环境能成功导入对应后端，返回 ``True``；否则返回 ``False``。
    """
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
    """返回请求的可用后端；未指定时返回所有可用的标准后端。

    Args:
        requested: 用户通过命令行显式指定的后端列表；为空时表示自动探测。

    Returns:
        实际可运行的后端名称列表。
    """
    if requested:
        return [b for b in requested if check_backend_available(b)]
    all_bk = ["numpy", "torch", "mlx", "jax"]
    return [b for b in all_bk if check_backend_available(b)]


def _enable_jax_x64():
    """启用 JAX x64，使 JAX 与其他后端保持相同的数据类型假设。

    Returns:
        无返回值。JAX 可用时会修改其全局配置；JAX 不可用时直接跳过。
    """
    if check_backend_available("jax"):
        import jax
        jax.config.update("jax_enable_x64", True)


def _format_ops(ops, unit="M"):
    """把吞吐量格式化成更容易阅读的 K/M 单位。

    Args:
        ops: 每秒操作数。
        unit: 格式化单位。``"M"`` 表示百万，``"K"`` 表示千，其他值表示不缩放。

    Returns:
        已格式化的吞吐量字符串。
    """
    if unit == "M":
        return f"{ops:.1f}M"
    if unit == "K":
        return f"{ops:.0f}K"
    return f"{ops:.0f}"


def _print_header(title):
    """打印单个测试组的标题分隔线。

    Args:
        title: 要显示在分隔线中间的标题。

    Returns:
        无返回值。函数只负责向标准输出打印内容。
    """
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_result(backend, elapsed, label, n_ops=None):
    """打印单条测试结果。

    n_ops 表示该测试中执行的逻辑操作数；传入后会额外计算 ops/s。

    Args:
        backend: 当前测试后端名称。
        elapsed: 测试耗时，单位为秒。
        label: 测试项中文标签。
        n_ops: 可选的逻辑操作数；传入后用于计算吞吐量。

    Returns:
        无返回值。函数只负责向标准输出打印内容。
    """
    if n_ops is not None:
        ops = n_ops / elapsed
        if ops > 1e6:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'M')} ops/s  |  {label}")
        else:
            print(f"  {backend:12s}  {elapsed:8.3f}s  |  {_format_ops(ops, 'K')} ops/s  |  {label}")
    else:
        print(f"  {backend:12s}  {elapsed:8.3f}s  |  {label}")


def run_benchmark(backend, workload, skip_groups):
    """对单个后端运行所有启用的基准测试组。

    Args:
        backend: 当前要测试的后端名称。
        workload: 负载适配器实例，提供实体池、关系表和模型负载测试方法。
        skip_groups: 要跳过的测试组集合，可包含 ``"A"``、``"B"``、``"C"``。

    Returns:
        当前后端的测试结果映射，键为测试项编号，值为 ``(标签, 耗时, 操作数)``。
    """
    print(f"\n{'─' * 70}")
    print(f"  后端: {backend}")
    print(f"{'─' * 70}")

    # all_results 收集当前后端所有未跳过测试组的结果，最终交给 print_summary 横向比较。
    all_results = {}

    if "A" not in skip_groups:
        # A 组只覆盖实体池操作，输出的 pool 会被后续关系表和模型负载复用。
        _print_header(f"测试组 A — 实体池 ({workload.iterations} 次)")
        pool, res_a = workload.bench_entity_pool(backend)
        all_results.update(res_a)
        for _, (label, elapsed, n_ops) in res_a.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        # 即使跳过 A 组，B/C 组仍然需要实体池作为关系端点。
        pool = workload.create_entity_pool(backend)

    if "B" not in skip_groups:
        # B 组创建并压测关系表；C 组会继续使用这里得到的两张关系表。
        _print_header(f"测试组 B — 关系表 ({workload.iterations} 次)")
        bb_rel, bf_rel, firm_pool, res_b = workload.bench_relation(backend, pool)
        all_results.update(res_b)
        for _, (label, elapsed, n_ops) in res_b.items():
            _print_result(backend, elapsed, label, n_ops)
    else:
        # 跳过 B 组时仍需构造关系表，否则完整模型负载没有输入网络。
        bb_rel, bf_rel, firm_pool = workload.create_relations(backend, pool)

    if "C" not in skip_groups:
        # C 组把实体池和关系表组合成业务模型，压测更接近真实应用的端到端负载。
        _print_header(f"测试组 C — 模型负载 ({workload.mc_runs} 条 MC × {workload.iterations} 轮)")
        res_c = workload.bench_model(pool, bb_rel, bf_rel, firm_pool)
        all_results.update(res_c)
        for _, (label, elapsed, n_ops) in res_c.items():
            _print_result(backend, elapsed, label, n_ops)

    return all_results


def print_summary(all_backend_results, skip_groups):
    """打印跨后端汇总结果。

    Args:
        all_backend_results: 后端名称到测试结果映射的字典。
        skip_groups: 被跳过的测试组集合。

    Returns:
        无返回值。函数只负责向标准输出打印汇总表。
    """
    print(f"\n\n{'=' * 70}")
    print("  跨后端性能汇总")
    print(f"{'=' * 70}")

    # test_items 是所有后端实际产生过的测试项合集，用于确定汇总表需要打印哪些行。
    test_items = set()
    for results in all_backend_results.values():
        test_items.update(results.keys())

    # 测试项编号的首字母就是组名，例如 A1_add 属于 A 组。
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
    """解析命令行参数。

    Returns:
        ``argparse.Namespace``，包含负载名称、规模、密度、迭代次数、后端和跳过组等配置。

    Examples:
        命令行运行示例::

            python demos/demo_multi_backend.py --backend numpy --iterations 10
    """
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
    """运行多后端基准测试命令行入口。

    Returns:
        无返回值。函数会解析命令行参数、运行基准测试并打印汇总结果。
    """
    args = parse_args()
    # backends 是实际会执行的后端列表；不可导入的后端会在 _get_available_backends 中被过滤。
    backends = _get_available_backends(args.backends)
    # skip_groups 控制 A/B/C 三组测试的执行范围，使用集合便于后续 O(1) 判断。
    skip_groups = set(args.skip_groups)
    # workload_cls 是业务负载类；运行器通过统一接口调用它，不依赖具体经济模型细节。
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
        # 每个后端运行前启用 JAX x64；非 JAX 后端会直接跳过。
        _enable_jax_x64()
        all_results[bk] = run_benchmark(bk, workload, skip_groups)

    print_summary(all_results, skip_groups)

    print(f"\n{'=' * 70}")
    print("  所有基准测试完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
