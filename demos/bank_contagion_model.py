"""
Reusable bank contagion model for RECS demos and benchmarks.

The model has two contagion channels:
- common-asset / common-borrower losses through bank-firm exposures;
- interbank default losses through creditor -> debtor edges.

Edge conventions:
- bank_firm_rel: src = bank, dst = firm/asset, amount = exposure.
- bank_bank_rel: src = creditor bank, dst = debtor bank, amount = loan exposure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from recs import RECS, Relation  # noqa: E402


@dataclass
class BankContagionParams:
    """Parameters for the bank contagion simulation."""

    external_shock_fraction: float = 0.05
    external_loss_low: float = 0.20
    external_loss_high: float = 0.50
    max_cascade_rounds: int = 8
    fire_sale_kappa: float = 0.25
    liquidation_rule: str = "interbank_first"
    reset_every: int = 10
    target_capital_ratio_low: float = 0.06
    target_capital_ratio_high: float = 0.12
    deposit_floor: float = 0.0
    eps: float = 1e-8
    seed: Optional[int] = None


@dataclass
class ContagionRunMetrics:
    """Compact metrics returned by one simulation run."""

    macro_steps: int = 0
    cascade_rounds: int = 0
    external_asset_loss: float = 0.0
    fire_sale_loss: float = 0.0
    interbank_loss: float = 0.0
    defaulted_banks: int = 0
    newly_defaulted_banks: int = 0

    def merge(self, other: "ContagionRunMetrics") -> None:
        self.macro_steps += other.macro_steps
        self.cascade_rounds += other.cascade_rounds
        self.external_asset_loss += other.external_asset_loss
        self.fire_sale_loss += other.fire_sale_loss
        self.interbank_loss += other.interbank_loss
        self.defaulted_banks = other.defaulted_banks
        self.newly_defaulted_banks += other.newly_defaulted_banks


@dataclass
class BankContagionScenarioData:
    """Backend-neutral random scenario arrays shared by all benchmark backends."""

    n_banks: int
    n_firms: int
    bank_risk_weight: np.ndarray
    firm_production: np.ndarray
    firm_da: np.ndarray
    initial_capital_ratio: np.ndarray
    ib_creditor: np.ndarray
    ib_debtor: np.ndarray
    ib_amount: np.ndarray
    bf_bank: np.ndarray
    bf_firm: np.ndarray
    bf_amount: np.ndarray
    bf_maturity: np.ndarray
    bf_rate: np.ndarray


@dataclass
class BankContagionShockPlan:
    """Pre-generated random shocks for one Monte Carlo path."""

    asset_loss_rates: np.ndarray
    reset_capital_ratios: np.ndarray


class BankContagionModel:
    """Balance-sheet contagion model backed by RECS entity and relation tables."""

    def __init__(
        self,
        bank_pool: RECS,
        firm_pool: RECS,
        bank_bank_rel: Relation,
        bank_firm_rel: Relation,
        params: Optional[BankContagionParams] = None,
        rng: Optional[np.random.Generator] = None,
        shock_plan: Optional[BankContagionShockPlan] = None,
    ):
        self.bank_pool = bank_pool
        self.firm_pool = firm_pool
        self.bank_bank_rel = bank_bank_rel
        self.bank_firm_rel = bank_firm_rel
        self.params = params or BankContagionParams()
        self.rng = rng or np.random.default_rng(self.params.seed)
        self.shock_plan = shock_plan

        self.n_banks = int(bank_pool.size)
        self.n_firms = int(firm_pool.size)
        self.bank_uid_to_pos = Relation.build_uid_to_pos(bank_pool.backend.to_numpy(bank_pool.uid_array()))
        self.firm_uid_to_pos = Relation.build_uid_to_pos(firm_pool.backend.to_numpy(firm_pool.uid_array()))

        self._ensure_state_columns()
        self._load_relation_state()
        initial_ratio = None
        if self.shock_plan is not None and self.shock_plan.reset_capital_ratios.size > 0:
            initial_ratio = self.shock_plan.reset_capital_ratios[0]
        self.reset_financial_state(reset_defaults=True, capital_ratio=initial_ratio)

    def _to_numpy(self, arr):
        return self.bank_pool.backend.to_numpy(arr)

    def _ensure_state_columns(self) -> None:
        for name in ("bankrupt", "propagated"):
            if name not in self.bank_pool.d:
                self.bank_pool.add_attribute(name, np.bool_, default=False)

    def _map_uids(self, uids, uid_to_pos, expected_size: int) -> np.ndarray:
        arr = np.asarray(uids, dtype=np.int64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        valid = (arr >= 0) & (arr < len(uid_to_pos))
        if np.any(valid):
            out[valid] = uid_to_pos[arr[valid]]
        if np.any((out < 0) | (out >= expected_size)):
            raise ValueError("Relation contains uid values that are not present in the target pool.")
        return out

    def _load_relation_state(self) -> None:
        bb_src_uid = self.bank_bank_rel.backend.to_numpy(self.bank_bank_rel.get_attr("src_uid"))
        bb_dst_uid = self.bank_bank_rel.backend.to_numpy(self.bank_bank_rel.get_attr("dst_uid"))
        bf_src_uid = self.bank_firm_rel.backend.to_numpy(self.bank_firm_rel.get_attr("src_uid"))
        bf_dst_uid = self.bank_firm_rel.backend.to_numpy(self.bank_firm_rel.get_attr("dst_uid"))

        self.bb_src = self._map_uids(bb_src_uid, self.bank_uid_to_pos, self.n_banks)
        self.bb_dst = self._map_uids(bb_dst_uid, self.bank_uid_to_pos, self.n_banks)
        self.bf_src = self._map_uids(bf_src_uid, self.bank_uid_to_pos, self.n_banks)
        self.bf_dst = self._map_uids(bf_dst_uid, self.firm_uid_to_pos, self.n_firms)

        self.initial_bb_amount = self.bank_bank_rel.backend.to_numpy(
            self.bank_bank_rel.get_attr("amount")
        ).astype(np.float64, copy=True)
        self.initial_bf_amount = self.bank_firm_rel.backend.to_numpy(
            self.bank_firm_rel.get_attr("amount")
        ).astype(np.float64, copy=True)

        self.bb_amount = self.initial_bb_amount.copy()
        self.bf_amount = self.initial_bf_amount.copy()

    def _row_sum(self, rows: np.ndarray, weights: np.ndarray, n_rows: int) -> np.ndarray:
        if rows.size == 0:
            return np.zeros(n_rows, dtype=np.float64)
        return np.bincount(rows, weights=weights, minlength=n_rows).astype(np.float64, copy=False)

    def _col_sum(self, cols: np.ndarray, weights: np.ndarray, n_cols: int) -> np.ndarray:
        if cols.size == 0:
            return np.zeros(n_cols, dtype=np.float64)
        return np.bincount(cols, weights=weights, minlength=n_cols).astype(np.float64, copy=False)

    def _bankrupt_mask(self) -> np.ndarray:
        return self._to_numpy(self.bank_pool.get_attr("bankrupt")).astype(bool)

    def _active_mask(self) -> np.ndarray:
        return ~self._bankrupt_mask()

    def _revalue_balance_sheet(self, preserve_deposits: bool = True, capital_ratio: Optional[np.ndarray] = None) -> None:
        a_exb = self._row_sum(self.bf_src, self.bf_amount, self.n_banks)
        a_ib = self._row_sum(self.bb_src, self.bb_amount, self.n_banks)
        z_ib = self._col_sum(self.bb_dst, self.bb_amount, self.n_banks)

        if preserve_deposits and "Z_deposit" in self.bank_pool.d:
            z_deposit = self._to_numpy(self.bank_pool.get_attr("Z_deposit")).astype(np.float64, copy=True)
        else:
            total_assets = a_exb + a_ib
            if capital_ratio is None:
                capital_ratio = self.rng.uniform(
                    self.params.target_capital_ratio_low,
                    self.params.target_capital_ratio_high,
                    self.n_banks,
                )
            else:
                capital_ratio = np.asarray(capital_ratio, dtype=np.float64)
                if capital_ratio.shape[0] != self.n_banks:
                    raise ValueError("capital_ratio length must equal n_banks")
            target_capital = total_assets * capital_ratio
            z_deposit = np.maximum(total_assets - z_ib - target_capital, self.params.deposit_floor)

        capital = (a_exb + a_ib - z_ib - z_deposit).astype(np.float32)
        self.bank_pool.assign(
            np.arange(self.n_banks),
            {
                "A_exb": a_exb.astype(np.float32),
                "A_IB": a_ib.astype(np.float32),
                "Z_IB": z_ib.astype(np.float32),
                "Z_deposit": z_deposit.astype(np.float32),
                "capital": capital,
                "equity": capital.copy(),
            },
        )

    def reset_financial_state(self, reset_defaults: bool = True, capital_ratio: Optional[np.ndarray] = None) -> None:
        """Reset working exposures from relation tables and rebuild the balance sheet."""
        self.bb_amount = self.initial_bb_amount.copy()
        self.bf_amount = self.initial_bf_amount.copy()
        self._revalue_balance_sheet(preserve_deposits=False, capital_ratio=capital_ratio)
        if reset_defaults:
            idx = np.arange(self.n_banks)
            self.bank_pool.assign(idx, {"bankrupt": np.zeros(self.n_banks, dtype=np.bool_),
                                        "propagated": np.zeros(self.n_banks, dtype=np.bool_)})
            self.bank_pool.enable(idx)

    def apply_common_asset_shock(self, asset_loss_rates: Optional[np.ndarray] = None) -> float:
        """Apply an exogenous shock to a random firm/asset set and return bank losses."""
        active_banks = self._active_mask()
        if self.n_firms == 0 or self.bf_amount.size == 0 or not np.any(active_banks):
            return 0.0

        if asset_loss_rates is None:
            firm_shock = self.rng.random(self.n_firms) < self.params.external_shock_fraction
            if not np.any(firm_shock):
                return 0.0
            loss_rates = np.zeros(self.n_firms, dtype=np.float64)
            loss_rates[firm_shock] = self.rng.uniform(
                self.params.external_loss_low,
                self.params.external_loss_high,
                int(firm_shock.sum()),
            )
        else:
            loss_rates = np.asarray(asset_loss_rates, dtype=np.float64)
            if loss_rates.shape[0] != self.n_firms:
                raise ValueError("asset_loss_rates length must equal n_firms")
            if not np.any(loss_rates > 0):
                return 0.0

        edge_loss_rate = loss_rates[self.bf_dst]
        edge_mask = (edge_loss_rate > 0) & active_banks[self.bf_src]
        if not np.any(edge_mask):
            return 0.0

        losses = self.bf_amount[edge_mask] * edge_loss_rate[edge_mask]
        self.bf_amount[edge_mask] = np.maximum(self.bf_amount[edge_mask] - losses, 0.0)
        self._revalue_balance_sheet(preserve_deposits=True)
        return float(losses.sum())

    def _new_default_indices(self) -> np.ndarray:
        capital = self._to_numpy(self.bank_pool.get_attr("capital")).astype(np.float64)
        bankrupt = self._bankrupt_mask()
        return np.nonzero((capital < -self.params.eps) & (~bankrupt))[0]

    def _interbank_default_amounts(self, defaulted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        capital = self._to_numpy(self.bank_pool.get_attr("capital")).astype(np.float64)
        z_ib = self._to_numpy(self.bank_pool.get_attr("Z_IB")).astype(np.float64)
        z_deposit = self._to_numpy(self.bank_pool.get_attr("Z_deposit")).astype(np.float64)

        gap = np.maximum(-capital[defaulted], 0.0)
        if self.params.liquidation_rule == "pari_passu":
            z_all = z_ib[defaulted] + z_deposit[defaulted]
            share = np.divide(z_ib[defaulted], z_all, out=np.zeros_like(gap), where=z_all > self.params.eps)
            d_ib = np.minimum(gap * share, z_ib[defaulted])
        elif self.params.liquidation_rule == "interbank_first":
            d_ib = np.minimum(gap, z_ib[defaulted])
        else:
            raise ValueError("liquidation_rule must be 'interbank_first' or 'pari_passu'")
        return z_ib[defaulted], d_ib

    def _propagate_interbank_losses(self, defaulted: np.ndarray) -> float:
        if defaulted.size == 0 or self.bb_amount.size == 0:
            return 0.0

        z_ib_defaulted, d_ib = self._interbank_default_amounts(defaulted)
        default_loss_rate = np.divide(
            d_ib,
            z_ib_defaulted,
            out=np.zeros_like(d_ib),
            where=z_ib_defaulted > self.params.eps,
        )
        loss_rate_by_bank = np.zeros(self.n_banks, dtype=np.float64)
        loss_rate_by_bank[defaulted] = default_loss_rate

        active_recipient = self._active_mask()
        active_recipient[defaulted] = False
        edge_mask = (loss_rate_by_bank[self.bb_dst] > 0) & active_recipient[self.bb_src]
        if not np.any(edge_mask):
            return 0.0

        losses = self.bb_amount[edge_mask] * loss_rate_by_bank[self.bb_dst[edge_mask]]
        self.bb_amount[edge_mask] = np.maximum(self.bb_amount[edge_mask] - losses, 0.0)
        self._revalue_balance_sheet(preserve_deposits=True)
        return float(losses.sum())

    def _propagate_fire_sale_losses(self, defaulted: np.ndarray) -> float:
        if self.params.fire_sale_kappa <= 0 or defaulted.size == 0 or self.bf_amount.size == 0:
            return 0.0

        sale_edges = np.isin(self.bf_src, defaulted)
        if not np.any(sale_edges):
            return 0.0

        total_by_firm = self._col_sum(self.bf_dst, self.bf_amount, self.n_firms)
        sale_by_firm = self._col_sum(self.bf_dst[sale_edges], self.bf_amount[sale_edges], self.n_firms)
        pressure = np.divide(
            sale_by_firm,
            total_by_firm,
            out=np.zeros_like(sale_by_firm),
            where=total_by_firm > self.params.eps,
        )
        price_drop = 1.0 - np.exp(-self.params.fire_sale_kappa * pressure)

        active_recipient = self._active_mask()
        active_recipient[defaulted] = False
        edge_drop = price_drop[self.bf_dst]
        loss_edges = (edge_drop > 0) & active_recipient[self.bf_src]
        if not np.any(loss_edges):
            return 0.0

        losses = self.bf_amount[loss_edges] * edge_drop[loss_edges]
        self.bf_amount[loss_edges] = np.maximum(self.bf_amount[loss_edges] - losses, 0.0)
        self._revalue_balance_sheet(preserve_deposits=True)
        return float(losses.sum())

    def mark_bankrupt(self, defaulted: np.ndarray) -> None:
        if defaulted.size == 0:
            return
        self.bank_pool.assign(
            defaulted,
            {
                "bankrupt": np.ones(defaulted.size, dtype=np.bool_),
                "propagated": np.ones(defaulted.size, dtype=np.bool_),
            },
        )
        self.bank_pool.disable(defaulted)

    def run_macro_step(self, asset_loss_rates: Optional[np.ndarray] = None) -> ContagionRunMetrics:
        """Run one external shock and its endogenous cascade."""
        metrics = ContagionRunMetrics(macro_steps=1)
        metrics.external_asset_loss = self.apply_common_asset_shock(asset_loss_rates=asset_loss_rates)

        for _ in range(self.params.max_cascade_rounds):
            new_defaulted = self._new_default_indices()
            if new_defaulted.size == 0:
                break

            metrics.cascade_rounds += 1
            metrics.newly_defaulted_banks += int(new_defaulted.size)
            self.mark_bankrupt(new_defaulted)
            metrics.interbank_loss += self._propagate_interbank_losses(new_defaulted)
            metrics.fire_sale_loss += self._propagate_fire_sale_losses(new_defaulted)

        metrics.defaulted_banks = int(self._bankrupt_mask().sum())
        return metrics

    def run(self, n_steps: int) -> ContagionRunMetrics:
        """Run multiple macro steps, optionally resetting the scenario periodically."""
        total = ContagionRunMetrics()
        for step in range(int(n_steps)):
            if self.params.reset_every > 0 and step > 0 and step % self.params.reset_every == 0:
                reset_ratio = None
                if self.shock_plan is not None and step < self.shock_plan.reset_capital_ratios.shape[0]:
                    reset_ratio = self.shock_plan.reset_capital_ratios[step]
                self.reset_financial_state(reset_defaults=True, capital_ratio=reset_ratio)
            asset_loss_rates = None
            if self.shock_plan is not None:
                if step >= self.shock_plan.asset_loss_rates.shape[0]:
                    raise ValueError("shock_plan does not contain enough macro steps")
                asset_loss_rates = self.shock_plan.asset_loss_rates[step]
            total.merge(self.run_macro_step(asset_loss_rates=asset_loss_rates))
        return total


def initial_balance_sheet_from_scenario(
    scenario: BankContagionScenarioData,
    params: Optional[BankContagionParams] = None,
    capital_ratio: Optional[np.ndarray] = None,
) -> dict:
    """Compute initial bank balance-sheet arrays from backend-neutral scenario data."""
    params = params or BankContagionParams()
    ratio = scenario.initial_capital_ratio if capital_ratio is None else np.asarray(capital_ratio, dtype=np.float64)
    a_exb = np.bincount(scenario.bf_bank, weights=scenario.bf_amount, minlength=scenario.n_banks).astype(np.float64)
    a_ib = np.bincount(scenario.ib_creditor, weights=scenario.ib_amount, minlength=scenario.n_banks).astype(np.float64)
    z_ib = np.bincount(scenario.ib_debtor, weights=scenario.ib_amount, minlength=scenario.n_banks).astype(np.float64)
    total_assets = a_exb + a_ib
    target_capital = total_assets * ratio
    z_deposit = np.maximum(total_assets - z_ib - target_capital, params.deposit_floor)
    capital = total_assets - z_ib - z_deposit
    return {
        "A_exb": a_exb.astype(np.float32),
        "A_IB": a_ib.astype(np.float32),
        "Z_IB": z_ib.astype(np.float32),
        "Z_deposit": z_deposit.astype(np.float32),
        "capital": capital.astype(np.float32),
        "equity": capital.astype(np.float32),
    }


def generate_scenario_data(
    n_banks: int,
    n_firms: int,
    density: float,
    seed: Optional[int] = None,
    params: Optional[BankContagionParams] = None,
) -> BankContagionScenarioData:
    """Generate backend-neutral random arrays for one benchmark scenario."""
    params = params or BankContagionParams(seed=seed)
    rng = np.random.default_rng(seed)

    bank_risk_weight = rng.uniform(0.3, 1.0, n_banks).astype(np.float32)
    firm_production = rng.uniform(0.5, 2.0, n_firms).astype(np.float32)
    firm_da = rng.uniform(100, 5000, n_firms).astype(np.float32)
    initial_capital_ratio = rng.uniform(
        params.target_capital_ratio_low,
        params.target_capital_ratio_high,
        n_banks,
    ).astype(np.float64)

    n_edges_ib = max(1, int(n_banks * n_banks * density))
    ib_creditor = rng.integers(0, n_banks, n_edges_ib, dtype=np.int64)
    ib_debtor = rng.integers(0, n_banks, n_edges_ib, dtype=np.int64)
    keep = ib_creditor != ib_debtor
    ib_creditor = ib_creditor[keep]
    ib_debtor = ib_debtor[keep]
    ib_amount = rng.uniform(10, 1000, ib_creditor.size).astype(np.float32)

    n_edges_bf = max(1, int(n_banks * n_firms * density))
    bf_bank = rng.integers(0, n_banks, n_edges_bf, dtype=np.int64)
    bf_firm = rng.integers(0, n_firms, n_edges_bf, dtype=np.int64)
    bf_amount = rng.uniform(50, 2000, n_edges_bf).astype(np.float32)
    bf_maturity = rng.integers(1, 365, n_edges_bf).astype(np.float32)
    bf_rate = rng.uniform(0.03, 0.12, n_edges_bf).astype(np.float32)

    return BankContagionScenarioData(
        n_banks=n_banks,
        n_firms=n_firms,
        bank_risk_weight=bank_risk_weight,
        firm_production=firm_production,
        firm_da=firm_da,
        initial_capital_ratio=initial_capital_ratio,
        ib_creditor=ib_creditor,
        ib_debtor=ib_debtor,
        ib_amount=ib_amount,
        bf_bank=bf_bank,
        bf_firm=bf_firm,
        bf_amount=bf_amount,
        bf_maturity=bf_maturity,
        bf_rate=bf_rate,
    )


def generate_monte_carlo_plans(
    n_runs: int,
    n_steps: int,
    n_banks: int,
    n_firms: int,
    seed: Optional[int] = None,
    params: Optional[BankContagionParams] = None,
) -> list[BankContagionShockPlan]:
    """Generate precomputed shock arrays for Monte Carlo paths."""
    params = params or BankContagionParams(seed=seed)
    rng = np.random.default_rng(seed)
    plans = []
    for _ in range(int(n_runs)):
        shock_mask = rng.random((n_steps, n_firms)) < params.external_shock_fraction
        raw_loss = rng.uniform(params.external_loss_low, params.external_loss_high, (n_steps, n_firms))
        asset_loss_rates = np.where(shock_mask, raw_loss, 0.0).astype(np.float64)
        reset_capital_ratios = rng.uniform(
            params.target_capital_ratio_low,
            params.target_capital_ratio_high,
            (n_steps + 1, n_banks),
        ).astype(np.float64)
        plans.append(BankContagionShockPlan(asset_loss_rates=asset_loss_rates,
                                            reset_capital_ratios=reset_capital_ratios))
    return plans


def build_model_from_scenario_data(
    scenario: BankContagionScenarioData,
    backend: str = "numpy",
    params: Optional[BankContagionParams] = None,
    shock_plan: Optional[BankContagionShockPlan] = None,
) -> BankContagionModel:
    """Build a backend-specific RECS model from shared scenario arrays."""
    params = params or BankContagionParams()
    bank_attr = {
        "name": "U20",
        "A_exb": np.float32,
        "A_IB": np.float32,
        "Z_IB": np.float32,
        "Z_deposit": np.float32,
        "capital": np.float32,
        "equity": np.float32,
        "risk_weight": np.float32,
    }
    firm_attr = {"name": "U20", "production": np.float32, "DA": np.float32}
    balance = initial_balance_sheet_from_scenario(
        scenario,
        params=params,
        capital_ratio=shock_plan.reset_capital_ratios[0] if shock_plan is not None else scenario.initial_capital_ratio,
    )

    bank_pool = RECS(capacity=scenario.n_banks, attr_dtypes=bank_attr, backend=backend)
    bank_pool.add(
        scenario.n_banks,
        name=[f"Bank_{i}" for i in range(scenario.n_banks)],
        A_exb=balance["A_exb"],
        A_IB=balance["A_IB"],
        Z_IB=balance["Z_IB"],
        Z_deposit=balance["Z_deposit"],
        capital=balance["capital"],
        equity=balance["equity"],
        risk_weight=scenario.bank_risk_weight,
    )

    firm_pool = RECS(capacity=scenario.n_firms, attr_dtypes=firm_attr, backend=backend)
    firm_pool.add(
        scenario.n_firms,
        name=[f"Firm_{i}" for i in range(scenario.n_firms)],
        production=scenario.firm_production,
        DA=scenario.firm_da,
    )

    bank_uids = bank_pool.backend.to_numpy(bank_pool.uid_array()).astype(np.int64)
    firm_uids = firm_pool.backend.to_numpy(firm_pool.uid_array()).astype(np.int64)

    bank_bank_rel = Relation(
        "interbank_credit",
        capacity=max(8, scenario.ib_amount.size),
        attr_dtypes={"amount": np.float32},
        backend=backend,
    )
    bank_bank_rel.add(
        bank_uids[scenario.ib_creditor],
        bank_uids[scenario.ib_debtor],
        amount=scenario.ib_amount,
    )

    bank_firm_rel = Relation(
        "firm_credit",
        capacity=max(8, scenario.bf_amount.size),
        attr_dtypes={"amount": np.float32, "maturity": np.float32, "rate": np.float32},
        backend=backend,
    )
    bank_firm_rel.add(
        bank_uids[scenario.bf_bank],
        firm_uids[scenario.bf_firm],
        amount=scenario.bf_amount,
        maturity=scenario.bf_maturity,
        rate=scenario.bf_rate,
    )

    return BankContagionModel(
        bank_pool,
        firm_pool,
        bank_bank_rel,
        bank_firm_rel,
        params=params,
        shock_plan=shock_plan,
    )


def create_contagion_scenario(
    n_banks: int,
    n_firms: int,
    density: float,
    backend: str = "numpy",
    seed: Optional[int] = None,
    params: Optional[BankContagionParams] = None,
) -> BankContagionModel:
    """Create a complete random scenario and return a ready-to-run model."""
    params = params or BankContagionParams(seed=seed)
    scenario = generate_scenario_data(n_banks, n_firms, density, seed=seed, params=params)
    return build_model_from_scenario_data(scenario, backend=backend, params=params)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reusable RECS bank contagion model.")
    parser.add_argument("--n-banks", type=int, default=5000)
    parser.add_argument("--n-firms", type=int, default=2000)
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--backend", type=str, default="numpy")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fire-sale-kappa", type=float, default=0.25)
    parser.add_argument("--liquidation-rule", choices=["interbank_first", "pari_passu"], default="interbank_first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = BankContagionParams(
        seed=args.seed,
        fire_sale_kappa=args.fire_sale_kappa,
        liquidation_rule=args.liquidation_rule,
    )
    model = create_contagion_scenario(
        n_banks=args.n_banks,
        n_firms=args.n_firms,
        density=args.density,
        backend=args.backend,
        seed=args.seed,
        params=params,
    )
    metrics = model.run(args.steps)
    print("RECS bank contagion model")
    print(f"params: {asdict(params)}")
    print(f"metrics: {asdict(metrics)}")


if __name__ == "__main__":
    main()
