"""
Bank contagion workload adapter for the generic multi-backend benchmark.

This module keeps workload-specific scenario construction, relation setup, and
model execution out of demos/demo_multi_backend.py.
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
    """Default workload used by the multi-backend benchmark."""

    name = "bank-contagion"
    display_name = "银行风险传染"

    def __init__(self, n_banks: int, n_firms: int, density: float, iterations: int, mc_runs: int, seed: int):
        self.n_banks = int(n_banks)
        self.n_firms = int(n_firms)
        self.density = float(density)
        self.iterations = int(iterations)
        self.mc_runs = int(mc_runs)
        self.seed = int(seed)
        self.params = BankContagionParams(
            seed=self.seed,
            max_cascade_rounds=8,
            reset_every=10,
            fire_sale_kappa=0.25,
            liquidation_rule="interbank_first",
        )
        self.scenario_data = generate_scenario_data(
            self.n_banks,
            self.n_firms,
            self.density,
            seed=self.seed,
            params=self.params,
        )
        self.mc_plans = generate_monte_carlo_plans(
            self.mc_runs,
            self.iterations,
            self.n_banks,
            self.n_firms,
            seed=self.seed + 1,
            params=self.params,
        )

    def config_lines(self) -> list[tuple[str, object]]:
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
        balance = initial_balance_sheet_from_scenario(self.scenario_data, params=self.params)
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
        results = {}
        t0 = time.time()
        pool = self.create_entity_pool(backend)
        t1 = time.time()
        results["A1_add"] = ("批量添加实体", t1 - t0, self.n_banks)

        t0 = time.time()
        for _ in range(self.iterations):
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
        results = {}

        t0 = time.time()
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
            order = bank_firm_rel.argsort_by("amount", ascending=False)
            bank_firm_rel.take(order[:min(100, bank_firm_rel.size)])
        t1 = time.time()
        results["B5_sort"] = ("排序与 Top-K", t1 - t0, self.iterations)

        t0 = time.time()
        for _ in range(min(self.iterations, 10)):
            if bank_firm_rel.size > 1:
                order_asc = bank_firm_rel.argsort_by("amount", ascending=True)
                n_remove = max(1, bank_firm_rel.size // 100)
                bank_firm_rel.remove_by_indices(order_asc[:n_remove])
        t1 = time.time()
        results["B6_remove"] = ("边删除", t1 - t0, min(self.iterations, 10))

        return bank_bank_rel, bank_firm_rel, firm_pool, results

    def bench_model(self, pool, bank_bank_rel, bank_firm_rel, firm_pool):
        results = {}
        t0 = time.time()
        model = None
        for plan in self.mc_plans:
            if model is None:
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
