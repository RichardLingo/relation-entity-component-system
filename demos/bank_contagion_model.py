"""
RECS 演示与基准测试复用的银行风险传染模型。

模型包含两类传染通道：
- 通过银行-厂商敞口产生的共同资产/共同借款人损失；
- 通过债权银行 -> 债务银行边产生的银行间违约损失。

边约定：
- bank_firm_rel：src = 银行，dst = 厂商/资产，amount = 敞口。
- bank_bank_rel：src = 债权银行，dst = 债务银行，amount = 贷款敞口。
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
    """银行风险传染仿真的参数。

    Attributes:
        external_shock_fraction: 每个宏观步骤中，单个厂商/资产被外生冲击命中的概率。
        external_loss_low: 被冲击资产损失率随机区间的下界。
        external_loss_high: 被冲击资产损失率随机区间的上界。
        max_cascade_rounds: 一次外生冲击后，最多迭代多少轮内生违约传染。
        fire_sale_kappa: 抛售压力转换为资产价格下跌的强度参数，越大表示折价越剧烈。
        liquidation_rule: 违约银行损失分摊规则。``interbank_first`` 表示银行间债务优先
            吸收缺口，``pari_passu`` 表示银行间债务与存款按比例共同吸收缺口。
        reset_every: 每隔多少个宏观步骤重置一次资产负债表；0 或负数表示不自动重置。
        target_capital_ratio_low: 初始化或重置时目标资本率随机区间的下界。
        target_capital_ratio_high: 初始化或重置时目标资本率随机区间的上界。
        deposit_floor: 存款负债的最小值，用于避免重估时出现负存款。
        eps: 数值容差，避免浮点误差导致边界判断不稳定。
        seed: 随机数种子；传入 ``None`` 时使用 NumPy 默认随机源。

    Examples:
        >>> params = BankContagionParams(seed=42, liquidation_rule="interbank_first")
        >>> params.max_cascade_rounds
        8
    """

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
    """单次仿真运行返回的紧凑指标。

    Attributes:
        macro_steps: 已经执行的宏观步骤数。
        cascade_rounds: 所有宏观步骤累计触发的级联传染轮数。
        external_asset_loss: 共同资产/共同借款人冲击造成的直接资产损失总额。
        fire_sale_loss: 违约银行抛售资产后，其他银行因价格下跌承受的损失总额。
        interbank_loss: 债务银行违约后，债权银行在银行间敞口上承受的损失总额。
        defaulted_banks: 当前累计违约银行数量，按最后一个宏观步骤结束时统计。
        newly_defaulted_banks: 本次运行中新发生违约的银行数量累计值。
    """

    macro_steps: int = 0
    cascade_rounds: int = 0
    external_asset_loss: float = 0.0
    fire_sale_loss: float = 0.0
    interbank_loss: float = 0.0
    defaulted_banks: int = 0
    newly_defaulted_banks: int = 0

    def merge(self, other: "ContagionRunMetrics") -> None:
        """把另一组运行指标累计到当前对象。

        Args:
            other: 待合并的单次运行指标。

        Returns:
            无返回值。当前对象会被原地更新。
        """
        self.macro_steps += other.macro_steps
        self.cascade_rounds += other.cascade_rounds
        self.external_asset_loss += other.external_asset_loss
        self.fire_sale_loss += other.fire_sale_loss
        self.interbank_loss += other.interbank_loss
        self.defaulted_banks = other.defaulted_banks
        self.newly_defaulted_banks += other.newly_defaulted_banks


@dataclass
class BankContagionScenarioData:
    """所有基准测试后端共享的后端无关随机场景数组。

    这些字段全部使用 NumPy 数组保存，是为了让同一份随机场景可以复用于 NumPy、
    PyTorch、MLX、JAX 等后端，避免不同后端各自抽样导致基准测试不可比。

    Attributes:
        n_banks: 银行实体数量；银行位置索引范围为 ``[0, n_banks)``。
        n_firms: 厂商/资产实体数量；厂商位置索引范围为 ``[0, n_firms)``。
        bank_risk_weight: 每家银行的风险权重，用于构造实体属性和查询压力条件。
        firm_production: 每个厂商的生产率属性，示例中作为厂商实体的普通属性保存。
        firm_da: 每个厂商的贷款需求或贷款规模属性，示例中作为厂商实体属性保存。
        initial_capital_ratio: 每家银行初始化资产负债表时使用的目标资本率。
        ib_creditor: 银行间关系的债权银行位置索引数组；与 ``ib_debtor``、
            ``ib_amount`` 一一对应。
        ib_debtor: 银行间关系的债务银行位置索引数组；每个元素指向一条边的目标银行。
        ib_amount: 银行间贷款敞口金额；第 k 项表示 ``ib_creditor[k]`` 对
            ``ib_debtor[k]`` 的敞口。
        bf_bank: 银行-厂商关系中的银行位置索引数组；与 ``bf_firm``、``bf_amount``
            等字段一一对应。
        bf_firm: 银行-厂商关系中的厂商位置索引数组；每个元素指向一条贷款资产边的目标厂商。
        bf_amount: 银行持有的贷款类资产敞口金额；第 k 项属于
            ``bf_bank[k] -> bf_firm[k]``。
        bf_maturity: 银行-厂商贷款边的剩余期限示例属性，单位可理解为天。
        bf_rate: 银行-厂商贷款边的利率示例属性。
    """

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
    """单条蒙特卡洛路径的预生成随机冲击。

    Attributes:
        asset_loss_rates: 形状为 ``(n_steps, n_firms)`` 的损失率矩阵。第 t 行表示
            第 t 个宏观步骤中每个厂商/资产的外生损失率；0 表示未被冲击。
        reset_capital_ratios: 形状为 ``(n_steps + 1, n_banks)`` 的资本率矩阵。
            第 0 行用于初始状态，后续行在周期性重置资产负债表时使用。
    """

    asset_loss_rates: np.ndarray
    reset_capital_ratios: np.ndarray


class BankContagionModel:
    """由 RECS 实体表和关系表支撑的资产负债表传染模型。

    模型把银行和厂商分别放在两个 RECS 实体池中，把金融敞口放在 Relation 边表中。
    银行实体保存资产负债表列，关系表保存贷款敞口列；运行时通过聚合边表金额重估
    每家银行的资产、负债和资本。

    Args:
        bank_pool: 银行实体池，至少需要包含资产、负债和资本相关属性列。
        firm_pool: 厂商/资产实体池，作为银行-厂商关系表的目标端点。
        bank_bank_rel: 银行间关系表，边属性 ``amount`` 表示银行间贷款敞口。
        bank_firm_rel: 银行-厂商关系表，边属性 ``amount`` 表示贷款类资产敞口。
        params: 模型参数。传入 ``None`` 时使用 ``BankContagionParams`` 默认值。
        rng: NumPy 随机数生成器。传入 ``None`` 时根据 ``params.seed`` 创建。
        shock_plan: 预生成冲击路径。传入后模型运行会复用该路径，适合跨后端基准测试。

    Raises:
        ValueError: 当关系表中存在无法映射到实体池的 uid 时抛出。
    """

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
        # Relation 边表保存的是稳定 uid；仿真中大量使用数组位置做聚合，因此先建立 uid 到位置的映射。
        self.bank_uid_to_pos = Relation.build_uid_to_pos(bank_pool.backend.to_numpy(bank_pool.uid_array()))
        self.firm_uid_to_pos = Relation.build_uid_to_pos(firm_pool.backend.to_numpy(firm_pool.uid_array()))

        self._ensure_state_columns()
        self._load_relation_state()
        initial_ratio = None
        if self.shock_plan is not None and self.shock_plan.reset_capital_ratios.size > 0:
            initial_ratio = self.shock_plan.reset_capital_ratios[0]
        self.reset_financial_state(reset_defaults=True, capital_ratio=initial_ratio)

    def _to_numpy(self, arr):
        """把当前后端数组转换为 NumPy 数组。

        Args:
            arr: 任意由 RECS 后端返回的数组。

        Returns:
            与输入数据对应的 NumPy 数组。
        """
        return self.bank_pool.backend.to_numpy(arr)

    def _ensure_state_columns(self) -> None:
        # bankrupt 表示银行是否已经违约；propagated 预留给更细粒度的传染状态追踪。
        for name in ("bankrupt", "propagated"):
            if name not in self.bank_pool.d:
                self.bank_pool.add_attribute(name, np.bool_, default=False)

    def _map_uids(self, uids, uid_to_pos, expected_size: int) -> np.ndarray:
        # 将 Relation 中的 uid 转换为连续位置索引，便于后续用 bincount 做高速聚合。
        arr = np.asarray(uids, dtype=np.int64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        valid = (arr >= 0) & (arr < len(uid_to_pos))
        if np.any(valid):
            out[valid] = uid_to_pos[arr[valid]]
        if np.any((out < 0) | (out >= expected_size)):
            raise ValueError("Relation contains uid values that are not present in the target pool.")
        return out

    def _load_relation_state(self) -> None:
        # 读取两张关系表的端点 uid，并转换成银行/厂商池内的位置索引。
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

        # initial_* 保留基准场景原值；工作副本会在冲击和传染过程中被逐步扣减。
        self.bb_amount = self.initial_bb_amount.copy()
        self.bf_amount = self.initial_bf_amount.copy()

    def _row_sum(self, rows: np.ndarray, weights: np.ndarray, n_rows: int) -> np.ndarray:
        """按行索引聚合权重。

        Args:
            rows: 每条边对应的源端位置索引。
            weights: 每条边参与求和的权重，通常是敞口金额。
            n_rows: 输出数组长度。

        Returns:
            长度为 ``n_rows`` 的行聚合结果。
        """
        if rows.size == 0:
            return np.zeros(n_rows, dtype=np.float64)
        return np.bincount(rows, weights=weights, minlength=n_rows).astype(np.float64, copy=False)

    def _col_sum(self, cols: np.ndarray, weights: np.ndarray, n_cols: int) -> np.ndarray:
        """按列索引聚合权重。

        Args:
            cols: 每条边对应的目标端位置索引。
            weights: 每条边参与求和的权重，通常是敞口金额。
            n_cols: 输出数组长度。

        Returns:
            长度为 ``n_cols`` 的列聚合结果。
        """
        if cols.size == 0:
            return np.zeros(n_cols, dtype=np.float64)
        return np.bincount(cols, weights=weights, minlength=n_cols).astype(np.float64, copy=False)

    def _bankrupt_mask(self) -> np.ndarray:
        """返回银行违约状态掩码。

        Returns:
            布尔数组；``True`` 表示对应银行已经违约。
        """
        return self._to_numpy(self.bank_pool.get_attr("bankrupt")).astype(bool)

    def _active_mask(self) -> np.ndarray:
        """返回银行仍可参与传染计算的状态掩码。

        Returns:
            布尔数组；``True`` 表示对应银行尚未违约。
        """
        return ~self._bankrupt_mask()

    def _revalue_balance_sheet(self, preserve_deposits: bool = True, capital_ratio: Optional[np.ndarray] = None) -> None:
        # A_exb 来自银行-厂商贷款边的行和；A_IB 来自银行间贷款边的行和；Z_IB 来自银行间贷款边的列和。
        a_exb = self._row_sum(self.bf_src, self.bf_amount, self.n_banks)
        a_ib = self._row_sum(self.bb_src, self.bb_amount, self.n_banks)
        z_ib = self._col_sum(self.bb_dst, self.bb_amount, self.n_banks)

        if preserve_deposits and "Z_deposit" in self.bank_pool.d:
            z_deposit = self._to_numpy(self.bank_pool.get_attr("Z_deposit")).astype(np.float64, copy=True)
        else:
            # 初始化或重置时用目标资本率反推存款负债，使资产负债表满足：
            # 资本 = 外部资产 + 银行间资产 - 银行间负债 - 存款负债。
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
        """从关系表重置工作敞口，并重建资产负债表。

        reset_defaults=True 时，会把所有银行重新标记为未违约且启用，适合蒙特卡洛路径之间
        复用同一个模型对象，减少反复构造实体池和关系表的开销。

        Args:
            reset_defaults: 是否同时清空违约状态并重新启用所有银行。
            capital_ratio: 可选的目标资本率数组，长度必须等于银行数量。传入 ``None`` 时现场抽样。

        Returns:
            无返回值。模型内部的工作敞口和银行实体池属性会被原地更新。

        Raises:
            ValueError: 当 ``capital_ratio`` 长度与银行数量不一致时抛出。
        """
        self.bb_amount = self.initial_bb_amount.copy()
        self.bf_amount = self.initial_bf_amount.copy()
        self._revalue_balance_sheet(preserve_deposits=False, capital_ratio=capital_ratio)
        if reset_defaults:
            idx = np.arange(self.n_banks)
            self.bank_pool.assign(idx, {"bankrupt": np.zeros(self.n_banks, dtype=np.bool_),
                                        "propagated": np.zeros(self.n_banks, dtype=np.bool_)})
            self.bank_pool.enable(idx)

    def apply_common_asset_shock(self, asset_loss_rates: Optional[np.ndarray] = None) -> float:
        """对随机厂商/资产集合施加外生冲击，并返回银行损失。

        如果传入 asset_loss_rates，则使用预生成冲击；否则现场抽样。损失只施加到仍处于
        active 状态的银行持有的银行-厂商敞口上。

        Args:
            asset_loss_rates: 可选的一维损失率数组，长度必须等于厂商/资产数量。
                第 j 项表示第 j 个厂商/资产在本步骤中的外生损失率。

        Returns:
            本次共同资产冲击造成的银行直接损失总额。

        Raises:
            ValueError: 当 ``asset_loss_rates`` 长度与厂商/资产数量不一致时抛出。
        """
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
        # 敞口扣减后立即重估资产负债表，后续违约判定会读取新的 capital 列。
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
            # 按比例清偿：银行间负债和存款负债共同承担资本缺口。
            z_all = z_ib[defaulted] + z_deposit[defaulted]
            share = np.divide(z_ib[defaulted], z_all, out=np.zeros_like(gap), where=z_all > self.params.eps)
            d_ib = np.minimum(gap * share, z_ib[defaulted])
        elif self.params.liquidation_rule == "interbank_first":
            # 银行间优先吸收：资本缺口先由银行间债权人承担，但不超过银行间负债总额。
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

        # 只有未违约的债权银行会继续承受损失；已违约银行不再重复计入接收方损失。
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
        # pressure 表示某个厂商/资产中由违约银行抛售的比例，随后用指数函数映射为价格跌幅。
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
        """把一批银行标记为违约并禁用。

        Args:
            defaulted: 新增违约银行的位置索引数组。

        Returns:
            无返回值。银行实体池中的 ``bankrupt``、``propagated`` 和启用状态会被原地更新。
        """
        if defaulted.size == 0:
            return
        # RECS 的启用/禁用状态可直接参与 include_active_only 查询；违约银行禁用后会被查询层过滤。
        self.bank_pool.assign(
            defaulted,
            {
                "bankrupt": np.ones(defaulted.size, dtype=np.bool_),
                "propagated": np.ones(defaulted.size, dtype=np.bool_),
            },
        )
        self.bank_pool.disable(defaulted)

    def run_macro_step(self, asset_loss_rates: Optional[np.ndarray] = None) -> ContagionRunMetrics:
        """运行一次外部冲击及其内生级联过程。

        一个宏观步骤先施加共同资产冲击，再循环处理“新增违约 -> 银行间损失 ->
        抛售损失 -> 重新判定违约”，直到没有新增违约或达到最大轮数。

        Args:
            asset_loss_rates: 可选的一维损失率数组。传入时使用指定冲击；不传时现场抽样。

        Returns:
            当前宏观步骤产生的损失、违约和级联轮数指标。

        Raises:
            ValueError: 当 ``asset_loss_rates`` 长度与厂商/资产数量不一致时抛出。
        """
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
        """运行多个宏观步骤，并可按周期重置场景。

        对基准测试来说，周期性重置可以避免所有银行在早期路径中全部退出，从而让每一轮
        仍然有足够的实体和边参与计算。

        Args:
            n_steps: 要运行的宏观步骤数。

        Returns:
            多个宏观步骤累计后的运行指标。

        Raises:
            ValueError: 当预生成冲击路径的步数少于 ``n_steps`` 时抛出。
        """
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
    """根据后端无关场景数据计算银行初始资产负债表数组。

    返回字典中的键会直接成为银行实体池的属性列：
    - A_exb：银行对厂商/外部资产的资产端敞口。
    - A_IB：银行作为债权人持有的银行间资产。
    - Z_IB：银行作为债务人形成的银行间负债。
    - Z_deposit：由目标资本率反推得到的存款负债。
    - capital / equity：示例中二者取同一数值，表示权益资本。

    Args:
        scenario: 后端无关场景数据，提供端点索引、敞口金额和初始资本率。
        params: 模型参数。传入 ``None`` 时使用默认参数。
        capital_ratio: 可选目标资本率数组。传入后覆盖 ``scenario.initial_capital_ratio``。

    Returns:
        字典形式的银行资产负债表列，键包括 ``A_exb``、``A_IB``、``Z_IB``、
        ``Z_deposit``、``capital`` 和 ``equity``。

    Examples:
        >>> scenario = generate_scenario_data(3, 2, 0.5, seed=1)
        >>> balance = initial_balance_sheet_from_scenario(scenario)
        >>> sorted(balance)
        ['A_IB', 'A_exb', 'Z_IB', 'Z_deposit', 'capital', 'equity']
    """
    params = params or BankContagionParams()
    # ratio 是每家银行的目标资本率；若调用者传入覆盖值，就用覆盖值重建资产负债表。
    ratio = scenario.initial_capital_ratio if capital_ratio is None else np.asarray(capital_ratio, dtype=np.float64)
    # a_exb、a_ib、z_ib 分别由边表端点数组聚合得到，是资产负债表的三个核心输入。
    a_exb = np.bincount(scenario.bf_bank, weights=scenario.bf_amount, minlength=scenario.n_banks).astype(np.float64)
    a_ib = np.bincount(scenario.ib_creditor, weights=scenario.ib_amount, minlength=scenario.n_banks).astype(np.float64)
    z_ib = np.bincount(scenario.ib_debtor, weights=scenario.ib_amount, minlength=scenario.n_banks).astype(np.float64)
    total_assets = a_exb + a_ib
    # target_capital 是按目标资本率要求保留的权益资本，存款负债由资产端扣除其他负债和资本后反推。
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
    """为单个基准测试场景生成后端无关随机数组。

    density 控制关系数量：银行间边大约为 n_banks * n_banks * density，
    银行-厂商边大约为 n_banks * n_firms * density。这里生成的是位置索引，
    构造 Relation 时再映射到各后端实体池实际分配的 uid。

    Args:
        n_banks: 银行实体数量。
        n_firms: 厂商/资产实体数量。
        density: 关系密度，控制银行间边和银行-厂商边的近似数量。
        seed: 随机种子。传入 ``None`` 时使用 NumPy 默认随机源。
        params: 模型参数。传入 ``None`` 时使用由 ``seed`` 初始化的默认参数。

    Returns:
        包含实体属性数组和关系端点数组的 ``BankContagionScenarioData``。

    Examples:
        >>> scenario = generate_scenario_data(10, 5, 0.1, seed=42)
        >>> scenario.n_banks, scenario.n_firms
        (10, 5)
    """
    params = params or BankContagionParams(seed=seed)
    rng = np.random.default_rng(seed)

    # bank_risk_weight 用于实体池查询示例；数值越低可理解为同等资本下风险承载更弱。
    bank_risk_weight = rng.uniform(0.3, 1.0, n_banks).astype(np.float32)
    # firm_production 和 firm_da 是厂商实体的普通业务属性，用来展示实体池可保存任意列。
    firm_production = rng.uniform(0.5, 2.0, n_firms).astype(np.float32)
    firm_da = rng.uniform(100, 5000, n_firms).astype(np.float32)
    # initial_capital_ratio 控制每家银行初始权益资本占总资产的比例。
    initial_capital_ratio = rng.uniform(
        params.target_capital_ratio_low,
        params.target_capital_ratio_high,
        n_banks,
    ).astype(np.float64)

    # 银行间敞口：creditor 是放款银行，debtor 是借款银行；去掉自借自贷边。
    n_edges_ib = max(1, int(n_banks * n_banks * density))
    ib_creditor = rng.integers(0, n_banks, n_edges_ib, dtype=np.int64)
    ib_debtor = rng.integers(0, n_banks, n_edges_ib, dtype=np.int64)
    keep = ib_creditor != ib_debtor
    ib_creditor = ib_creditor[keep]
    ib_debtor = ib_debtor[keep]
    ib_amount = rng.uniform(10, 1000, ib_creditor.size).astype(np.float32)

    # 银行-厂商敞口：bf_bank 持有对 bf_firm 的贷款类资产，并附带期限和利率两个示例属性。
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
    """生成蒙特卡洛路径使用的预计算冲击数组。

    预先生成冲击矩阵的目的，是让不同计算后端面对完全相同的随机路径；
    这样基准测试比较的是后端执行性能，而不是随机抽样差异。

    Args:
        n_runs: 蒙特卡洛路径数量。
        n_steps: 每条路径包含的宏观步骤数。
        n_banks: 银行实体数量，用于生成重置资本率矩阵。
        n_firms: 厂商/资产实体数量，用于生成资产冲击矩阵。
        seed: 随机种子。
        params: 模型参数。传入 ``None`` 时使用由 ``seed`` 初始化的默认参数。

    Returns:
        长度为 ``n_runs`` 的冲击路径列表。
    """
    params = params or BankContagionParams(seed=seed)
    rng = np.random.default_rng(seed)
    # plans 中每个元素代表一条完整的蒙特卡洛路径。
    plans = []
    for _ in range(int(n_runs)):
        # shock_mask 标记某个宏观步骤中哪些厂商/资产被外生冲击命中。
        shock_mask = rng.random((n_steps, n_firms)) < params.external_shock_fraction
        # raw_loss 保存被冲击资产的候选损失率，未命中的位置稍后会被置零。
        raw_loss = rng.uniform(params.external_loss_low, params.external_loss_high, (n_steps, n_firms))
        asset_loss_rates = np.where(shock_mask, raw_loss, 0.0).astype(np.float64)
        # reset_capital_ratios 的第 0 行用于初始状态，后续行用于周期性重置。
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
    """根据共享场景数组构建指定后端的 RECS 模型。

    该函数展示了从“后端无关的 NumPy 场景数组”到“指定后端 RECS 实体池/关系表”的完整转换：
    先创建银行池和厂商池，再把位置索引映射为池内 uid，最后创建两张 Relation 边表。

    Args:
        scenario: 后端无关场景数组。
        backend: 目标 RECS 后端名称，例如 ``numpy``、``torch``、``mlx`` 或 ``jax``。
        params: 模型参数。传入 ``None`` 时使用默认参数。
        shock_plan: 可选的预生成冲击路径。

    Returns:
        已完成实体池、关系表和初始资产负债表设置的 ``BankContagionModel``。

    Examples:
        >>> scenario = generate_scenario_data(10, 5, 0.1, seed=42)
        >>> model = build_model_from_scenario_data(scenario)
        >>> model.n_banks
        10
    """
    params = params or BankContagionParams()
    # 银行实体属性列：A_* 表示资产端项目，Z_* 表示负债端项目。
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
    # 厂商实体属性列：本 demo 只需要生产率和贷款需求两个业务属性。
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

    # Relation.add 接收的是实体 uid，不是位置索引；因此先取出实体池分配的 uid 数组。
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
    """创建完整随机场景，并返回可直接运行的模型。

    Args:
        n_banks: 银行实体数量。
        n_firms: 厂商/资产实体数量。
        density: 关系密度。
        backend: 目标 RECS 后端名称。
        seed: 随机种子。
        params: 模型参数。传入 ``None`` 时根据 ``seed`` 创建默认参数。

    Returns:
        可直接调用 ``run`` 的银行风险传染模型。
    """
    params = params or BankContagionParams(seed=seed)
    scenario = generate_scenario_data(n_banks, n_firms, density, seed=seed, params=params)
    return build_model_from_scenario_data(scenario, backend=backend, params=params)


def parse_args():
    """解析银行风险传染模型命令行参数。

    Returns:
        ``argparse.Namespace``，包含场景规模、网络密度、后端、步数和模型参数覆盖值。

    Examples:
        命令行运行示例::

            python demos/bank_contagion_model.py --n-banks 1000 --steps 5
    """
    parser = argparse.ArgumentParser(description="运行可复用的 RECS 银行风险传染模型。")
    parser.add_argument("--n-banks", type=int, default=5000,
                        help="银行实体数量（默认: 5000）")
    parser.add_argument("--n-firms", type=int, default=2000,
                        help="厂商/资产实体数量（默认: 2000）")
    parser.add_argument("--density", type=float, default=0.001,
                        help="关系网络密度，用于控制银行间边和银行-厂商边数量（默认: 0.001）")
    parser.add_argument("--backend", type=str, default="numpy",
                        help="计算后端名称，例如 numpy、torch、mlx 或 jax（默认: numpy）")
    parser.add_argument("--steps", type=int, default=10,
                        help="宏观仿真步骤数（默认: 10）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，用于复现场景和冲击路径（默认: 42）")
    parser.add_argument("--fire-sale-kappa", type=float, default=0.25,
                        help="抛售压力转换为资产价格下跌的强度参数（默认: 0.25）")
    parser.add_argument("--liquidation-rule", choices=["interbank_first", "pari_passu"], default="interbank_first",
                        help="违约损失分摊规则：interbank_first 为银行间优先吸收，pari_passu 为按比例分摊")
    return parser.parse_args()


def main() -> None:
    """运行银行风险传染模型命令行入口。

    Returns:
        无返回值。函数会创建随机场景、运行模型，并打印参数和指标字典。
    """
    args = parse_args()
    # params 汇集命令行可覆盖的模型参数；其余参数使用 BankContagionParams 默认值。
    params = BankContagionParams(
        seed=args.seed,
        fire_sale_kappa=args.fire_sale_kappa,
        liquidation_rule=args.liquidation_rule,
    )
    # model 已经包含银行池、厂商池和两张关系表，可直接调用 run 执行仿真。
    model = create_contagion_scenario(
        n_banks=args.n_banks,
        n_firms=args.n_firms,
        density=args.density,
        backend=args.backend,
        seed=args.seed,
        params=params,
    )
    # metrics 汇总了外生损失、银行间损失、抛售损失和违约数量等结果。
    metrics = model.run(args.steps)
    print("RECS bank contagion model")
    print(f"params: {asdict(params)}")
    print(f"metrics: {asdict(metrics)}")


if __name__ == "__main__":
    main()
