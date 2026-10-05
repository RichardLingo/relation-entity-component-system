"""
通用多后端基准测试使用的银行风险传染负载适配器。

本模块承载负载相关的场景构造、关系初始化和模型执行逻辑，
避免这些细节进入 demos/demo_multi_backend.py。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from recs import RECS, Relation  # noqa: E402

try:
    from bank_contagion_model import (
        BankContagionModel,
        BankContagionParams,
        generate_monte_carlo_plans,
        generate_scenario_data,
        initial_balance_sheet_from_scenario,
    )
except ModuleNotFoundError:
    from demos.bank_contagion_model import (  # noqa: E402
        BankContagionModel,
        BankContagionParams,
        generate_monte_carlo_plans,
        generate_scenario_data,
        initial_balance_sheet_from_scenario,
    )


class BankContagionBenchmarkWorkload:
    """多后端基准测试使用的默认负载。

    这个类把“银行风险传染”这个具体业务场景封装成统一接口，供
    demo_multi_backend.py 调用。运行器只关心 create_* 和 bench_* 方法，
    不需要知道银行、厂商、敞口和冲击路径的具体生成方式。

    Args:
        n_banks: 银行实体数量。
        n_firms: 厂商/资产实体数量。
        density: 关系密度，控制银行间关系和银行-厂商关系的规模。
        iterations: 每个基准测试项的迭代次数；模型负载中表示每条路径的宏观步骤数。
        mc_runs: 蒙特卡洛路径数量。
        seed: 场景生成和冲击路径生成使用的主随机种子。

    Attributes:
        name: 命令行和负载注册表使用的稳定名称。
        display_name: 输出给读者看的中文负载名称。
        scenario_data: 一次性生成、跨后端复用的随机场景数组。
        mc_plans: 一次性生成、跨后端复用的蒙特卡洛冲击路径。
    """

    name = "bank-contagion"
    display_name = "银行风险传染"

    def __init__(self, n_banks: int, n_firms: int, density: float, iterations: int, mc_runs: int, seed: int):
        # 命令行参数先统一转成确定类型，避免 argparse 或外部调用传入兼容但不稳定的类型。
        self.n_banks = int(n_banks)
        self.n_firms = int(n_firms)
        self.density = float(density)
        self.iterations = int(iterations)
        self.mc_runs = int(mc_runs)
        self.seed = int(seed)
        # 基准测试固定使用一组模型参数；随机部分由 seed 控制，便于复现实验结果。
        self.params = BankContagionParams(
            seed=self.seed,
            max_cascade_rounds=8,
            reset_every=10,
            fire_sale_kappa=0.25,
            liquidation_rule="interbank_first",
        )
        # 场景数据只生成一次，然后在不同后端之间复用，保证性能对比面对的是同一张网络。
        self.scenario_data = generate_scenario_data(
            self.n_banks,
            self.n_firms,
            self.density,
            seed=self.seed,
            params=self.params,
        )
        # 蒙特卡洛冲击路径也预先生成，避免不同后端各自抽样导致结果不可比较。
        self.mc_plans = generate_monte_carlo_plans(
            self.mc_runs,
            self.iterations,
            self.n_banks,
            self.n_firms,
            seed=self.seed + 1,
            params=self.params,
        )

    def config_lines(self) -> list[tuple[str, object]]:
        """返回用于打印配置摘要的键值对。

        Returns:
            二元组列表。每个二元组的第一项是显示标签，第二项是要打印的值。
        """
        return [
            ("负载", self.name),
            ("银行实体", f"{self.n_banks:,}"),
            ("厂商实体", f"{self.n_firms:,}"),
            ("关系密度", self.density),
            ("迭代次数", self.iterations),
            ("MC 路径", self.mc_runs),
            ("随机种子", self.seed),
        ]

    def create_entity_pool(self, backend: str):
        """创建银行实体池，并写入初始资产负债表属性。

        attr_types 定义实体表的列名和数据类型；pool.add 一次性批量添加 n_banks 个实体，
        每个关键字参数都会写入同名属性列。

        Args:
            backend: RECS 后端名称。

        Returns:
            已填充银行实体和资产负债表属性的 RECS 实体池。
        """
        balance = initial_balance_sheet_from_scenario(self.scenario_data, params=self.params)
        # attr_types 是银行实体表的列定义；键是列名，值是后端数组使用的数据类型。
        attr_types = {
            "name": "U20",
            "A_exb": np.float32,
            "A_IB": np.float32,
            "Z_IB": np.float32,
            "Z_deposit": np.float32,
            "capital": np.float32,
            "equity": np.float32,
            "risk_weight": np.float32,
        }
        pool = RECS(capacity=self.n_banks, attr_dtypes=attr_types, backend=backend)
        pool.add(
            self.n_banks,
            name=[f"Bank_{i}" for i in range(self.n_banks)],
            A_exb=balance["A_exb"],
            A_IB=balance["A_IB"],
            Z_IB=balance["Z_IB"],
            Z_deposit=balance["Z_deposit"],
            capital=balance["capital"],
            equity=balance["equity"],
            risk_weight=self.scenario_data.bank_risk_weight,
        )
        return pool

    def create_firm_pool(self, backend: str):
        """创建厂商实体池。

        厂商池在模型中既代表借款企业，也可理解为外部资产类别；银行-厂商关系表中的
        dst_uid 会指向这里的实体 uid。

        Args:
            backend: RECS 后端名称。

        Returns:
            已填充厂商实体属性的 RECS 实体池。
        """
        # firm_attr 是厂商实体表的列定义；DA 在这里表示贷款需求或贷款规模。
        firm_attr = {"name": "U20", "production": np.float32, "DA": np.float32}
        firm_pool = RECS(capacity=self.n_firms, attr_dtypes=firm_attr, backend=backend)
        firm_pool.add(
            self.n_firms,
            name=[f"Firm_{i}" for i in range(self.n_firms)],
            production=self.scenario_data.firm_production,
            DA=self.scenario_data.firm_da,
        )
        return firm_pool

    def bench_entity_pool(self, backend: str):
        """测试实体池常见操作：批量添加、条件查询、条件赋值和启用/禁用。

        Args:
            backend: RECS 后端名称。

        Returns:
            ``(pool, results)``。``pool`` 是后续测试继续复用的银行实体池；
            ``results`` 是测试项编号到 ``(标签, 耗时, 操作数)`` 的映射。
        """
        results = {}
        # results 保存统一格式的基准结果，便于 demo_multi_backend.py 汇总打印。
        t0 = time.time()
        pool = self.create_entity_pool(backend)
        t1 = time.time()
        results["A1_add"] = ("批量添加实体", t1 - t0, self.n_banks)

        t0 = time.time()
        for _ in range(self.iterations):
            # index 返回命中的实体位置，不复制属性列，适合只需要位置的高频筛选。
            pool.index(
                lambda p: (p.get_attr("capital") / (p.get_attr("risk_weight") + 1e-10)) < 500,
                return_="indices",
                include_active_only=True,
            )
            pool.index(
                lambda p: p.get_attr("Z_IB") > 1500,
                return_="indices",
                include_active_only=True,
            )
        t1 = time.time()
        results["A2_query"] = ("条件查询 (index)", t1 - t0, self.iterations * 2)

        t0 = time.time()
        for _ in range(self.iterations):
            # where_set 把“筛选”和“写入”合并成一个操作，适合批量修正满足条件的实体属性。
            pool.where_set(
                lambda p: p.get_attr("Z_IB") > 2000,
                {"capital": 500.0, "equity": 800.0},
                include_active_only=True,
            )
            pool.where_set(
                lambda p: p.get_attr("A_exb") < 1000,
                {"A_exb": 1000.0, "A_IB": 500.0},
                include_active_only=True,
            )
        t1 = time.time()
        results["A3_where_set"] = ("条件批量赋值 (where_set)", t1 - t0, self.iterations * 2)

        t0 = time.time()
        for _ in range(self.iterations):
            # 启用/禁用状态会影响 include_active_only=True 的查询结果。
            defaulted = pool.index(
                lambda p: p.get_attr("capital") < 0,
                return_="indices",
                include_active_only=True,
            )
            if len(defaulted) > 0:
                pool.disable(defaulted)
                pool.enable(defaulted[:max(1, len(defaulted) // 2)])
        t1 = time.time()
        results["A4_enable_disable"] = ("实体启用/禁用", t1 - t0, self.iterations)
        return pool, results

    def create_relations(self, backend: str, pool):
        """创建银行间关系表和银行-厂商关系表。

        银行间关系先用稠密矩阵演示 dense_matrix_to_relation；银行-厂商关系直接用
        Relation.add 批量添加边，展示从稀疏端点数组构造关系表的方式。

        Args:
            backend: RECS 后端名称。
            pool: 银行实体池，作为银行间关系两端和银行-厂商关系源端。

        Returns:
            ``(bank_bank_rel, bank_firm_rel, firm_pool)``，分别表示银行间关系表、
            银行-厂商关系表和厂商实体池。
        """
        # mat_ib[row, col] 表示 row 银行对 col 银行的债权敞口。
        mat_ib = np.zeros((self.n_banks, self.n_banks), dtype=np.float32)
        mat_ib[self.scenario_data.ib_creditor, self.scenario_data.ib_debtor] = self.scenario_data.ib_amount
        bank_bank_rel = pool.dense_matrix_to_relation(
            mat_ib,
            dst_pool=pool,
            relation_name="interbank",
            attr_name="amount",
            threshold=0.0,
        )

        firm_pool = self.create_firm_pool(backend)
        # 场景数据里的端点是位置索引，Relation 需要实体 uid，因此这里完成位置到 uid 的转换。
        bank_uids = pool.backend.to_numpy(pool.uid_array()).astype(np.int64)
        firm_uids = firm_pool.backend.to_numpy(firm_pool.uid_array()).astype(np.int64)
        bank_firm_rel = Relation(
            "credit",
            capacity=max(8, self.scenario_data.bf_amount.size),
            attr_dtypes={"amount": np.float32, "maturity": np.float32, "rate": np.float32},
            backend=backend,
        )
        bank_firm_rel.add(
            bank_uids[self.scenario_data.bf_bank],
            firm_uids[self.scenario_data.bf_firm],
            amount=self.scenario_data.bf_amount,
            maturity=self.scenario_data.bf_maturity,
            rate=self.scenario_data.bf_rate,
        )
        return bank_bank_rel, bank_firm_rel, firm_pool

    def bench_relation(self, backend: str, pool):
        """测试关系表常见操作：建边、查询、聚合、排序和删除。

        Args:
            backend: RECS 后端名称。
            pool: 银行实体池。

        Returns:
            ``(bank_bank_rel, bank_firm_rel, firm_pool, results)``。前三项供模型负载复用，
            ``results`` 保存 B 组各测试项的耗时和操作数。
        """
        results = {}

        t0 = time.time()
        # B1：从稠密矩阵转换成边表，常用于把传统邻接矩阵迁移到 Relation。
        mat_ib = np.zeros((self.n_banks, self.n_banks), dtype=np.float32)
        mat_ib[self.scenario_data.ib_creditor, self.scenario_data.ib_debtor] = self.scenario_data.ib_amount
        bank_bank_rel = pool.dense_matrix_to_relation(
            mat_ib,
            dst_pool=pool,
            relation_name="interbank",
            attr_name="amount",
            threshold=0.0,
        )
        t1 = time.time()
        results["B1_dense_to_rel"] = ("稠密→稀疏转换", t1 - t0, self.scenario_data.ib_amount.size)

        t0 = time.time()
        # B2：直接从 src_uid、dst_uid 和边属性数组批量创建稀疏关系。
        firm_pool = self.create_firm_pool(backend)
        bank_uids = pool.backend.to_numpy(pool.uid_array()).astype(np.int64)
        firm_uids = firm_pool.backend.to_numpy(firm_pool.uid_array()).astype(np.int64)
        bank_firm_rel = Relation(
            "credit",
            capacity=max(8, self.scenario_data.bf_amount.size),
            attr_dtypes={"amount": np.float32, "maturity": np.float32, "rate": np.float32},
            backend=backend,
        )
        bank_firm_rel.add(
            bank_uids[self.scenario_data.bf_bank],
            firm_uids[self.scenario_data.bf_firm],
            amount=self.scenario_data.bf_amount,
            maturity=self.scenario_data.bf_maturity,
            rate=self.scenario_data.bf_rate,
        )
        t1 = time.time()
        results["B2_add_edges"] = ("批量添加边", t1 - t0, self.scenario_data.bf_amount.size)

        t0 = time.time()
        for i in range(self.iterations):
            # B3：分别测试按端点 uid、按边属性、以及二者组合的索引查询。
            sample_uid = int(pool.d["i"][i % self.n_banks])
            bank_firm_rel.index(src_uid=sample_uid, return_="indices")
            bank_firm_rel.index(edge_pred=lambda r: r.get_attr("amount") > 500, return_="indices")
            bank_firm_rel.index(
                src_uid=sample_uid,
                edge_pred=lambda r: r.get_attr("amount") > 500,
                return_="indices",
            )
        t1 = time.time()
        results["B3_query"] = ("按 uid/属性查询", t1 - t0, self.iterations * 3)

        t0 = time.time()
        for _ in range(self.iterations):
            # B4：Relation 的行/列聚合等价于对邻接矩阵按行/列求和或计数，但避免保存完整稠密矩阵。
            bank_uid_to_pos = bank_firm_rel.build_uid_to_pos(pool.uid_array())
            firm_uid_to_pos = bank_firm_rel.build_uid_to_pos(firm_pool.uid_array())
            bank_firm_rel.row_sum("amount", bank_uid_to_pos, n_rows=self.n_banks)
            bank_firm_rel.col_sum("amount", firm_uid_to_pos, n_cols=self.n_firms)
            bank_firm_rel.row_count(bank_uid_to_pos, n_rows=self.n_banks)
            bank_firm_rel.col_count(firm_uid_to_pos, n_cols=self.n_firms)
        t1 = time.time()
        results["B4_agg"] = ("行和/列和聚合", t1 - t0, self.iterations * 4)

        t0 = time.time()
        for _ in range(self.iterations):
            # B5：按边属性排序后再 take，可实现 Top-K 风险敞口抽取。
            order = bank_firm_rel.argsort_by("amount", ascending=False)
            bank_firm_rel.take(order[:min(100, bank_firm_rel.size)])
        t1 = time.time()
        results["B5_sort"] = ("排序与 Top-K", t1 - t0, self.iterations)

        t0 = time.time()
        for _ in range(min(self.iterations, 10)):
            # B6：删除最小的一批边，用于模拟边表收缩或低权重关系裁剪。
            if bank_firm_rel.size > 1:
                order_asc = bank_firm_rel.argsort_by("amount", ascending=True)
                n_remove = max(1, bank_firm_rel.size // 100)
                bank_firm_rel.remove_by_indices(order_asc[:n_remove])
        t1 = time.time()
        results["B6_remove"] = ("边删除", t1 - t0, min(self.iterations, 10))

        return bank_bank_rel, bank_firm_rel, firm_pool, results

    def bench_model(self, pool, bank_bank_rel, bank_firm_rel, firm_pool):
        """测试完整银行风险传染模型负载。

        Args:
            pool: 银行实体池。
            bank_bank_rel: 银行间关系表。
            bank_firm_rel: 银行-厂商关系表。
            firm_pool: 厂商实体池。

        Returns:
            C 组测试结果映射，键为测试项编号，值为 ``(标签, 耗时, 操作数)``。
        """
        results = {}
        t0 = time.time()
        model = None
        for plan in self.mc_plans:
            if model is None:
                # 第一条路径需要创建模型；后续路径复用同一个模型对象，只重置状态和冲击计划。
                model = BankContagionModel(
                    pool,
                    firm_pool,
                    bank_bank_rel,
                    bank_firm_rel,
                    params=self.params,
                    shock_plan=plan,
                )
            else:
                model.shock_plan = plan
                model.reset_financial_state(reset_defaults=True, capital_ratio=plan.reset_capital_ratios[0])
            model.run(self.iterations)
        t1 = time.time()
        n_ops = self.iterations * max(1, len(self.mc_plans))
        results["C_workload"] = (f"模型负载（{self.name}, MC={len(self.mc_plans)}）", t1 - t0, n_ops)
        return results
