"""
@File   ?demo3.py
@Desc   : RECS demo: entity table and relation examples.
"""

import sys
from pathlib import Path

import numpy as np

try:
    from ecs.ecsengine import ECSEngine  # #NOTE 导入引擎模块。不能改动该行!
except ModuleNotFoundError:
    # 允许直接运行本 demo：把仓库根目录加入 sys.path
    # 这样在未设置 PYTHONPATH 的情况下也能运行。
    this_file = Path(__file__).resolve()
    # 结构：.../relation-entity-component-system/demos/demo3.py
    # 要能 import ecs.*，需要把“仓库根目录”加入 sys.path
    repo_root = this_file.parents[1]  # .../relation-entity-component-system
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ecs.ecsengine import ECSEngine  # noqa: E402

# from ecs.ecsengine import dense_to_relation
# from ecs.ecsengine import EntityPool

# 1. 初始化银行个体众，初始容量为5
attr_types_banks = {
    'name': 'U10',  # 银行名称，字符串类型，最大长度10
    'A_exb': np.float32,  # 对厂商的资产
    'A_IB_all': np.float32,  # 银间的资产
    'Z_IB_all': np.float32  # 银间的负债
}
et_banks = ECSEngine(capacity=5, attr_dtypes=attr_types_banks)
et_banks.d['o'][:] = True  # 启用所有银行实体
et_banks.d['name'][:] = ['Bank 1', 'Bank 2', 'Bank 3', 'Bank 4', 'Bank 5']  # 初始化银行名称
et_banks.d['A_exb'][:] = [1631.73, 4303.24, 3379.72, 4351.47, 620.69]  # 初始化对厂商的资产
et_banks.d['A_IB_all'][:] = [2185.24, 398.37, 730.99, 1357.75, 2717.39]  # 初始化银行间的资产
et_banks.d['Z_IB_all'][:] = [959.6, 1844.24, 1253.34, 2851.85, 480.71]  # 初始化银行间的负债
# 设置逻辑 size，使 pool 知道已有实体数量
et_banks._pool.size = 5
et_banks._pool.d['o'][:et_banks._pool.size] = True

# 1. 初始化厂商个体众，初始容量为4
attr_types_firms = {
    'name': 'U10',  # 厂商名称，字符串类型，最大长度10
    'production_rate': np.float32,  # 生产率
    'DA': np.float32  # 贷款金额
}
et_firms = ECSEngine(capacity=4, attr_dtypes=attr_types_firms)
et_firms.d['o'][:] = True  # 启用所有厂商实体
et_firms.d['name'][:] = ['Firm 1', 'Firm 2', 'Firm 3', 'Firm 4']  # 初始化厂商名称
et_firms.d['production_rate'][:] = [1.2, 0.9, 1.1, 1.3]  # 初始化生产率
et_firms.d['DA'][:] = [4361.83, 2575.97, 4259.21, 3089.84]  # 初始化贷款金额
# 设置逻辑 size
et_firms._pool.size = 4
et_firms._pool.d['o'][:et_firms._pool.size] = True

pass  #DEBUG

A_IB = np.array([
    [0, 1728.55, 0, 134.46, 322.23],
    [109.35, 0, 289.02, 0, 0],
    [730.99, 0, 0, 0, 0],
    [119.26, 115.69, 964.32, 0, 158.48],
    [0, 0, 0, 2717.39, 0]
], dtype=np.float32)  # 银行间拆借矩阵

# 将稠密的银行-银行矩阵转换为 Relation（基于 uid 的边表）
A_IB_rel = et_banks.dense_matrix_to_relation(A_IB, dst_pool=et_banks, relation_name='A_IB', attr_name='amount', threshold=0.0)

IBA = np.array([
    [685.08, 115.72, 229.3, 601.64],
    [809.53, 1163.32, 2052.77, 277.62],
    [1654.57, 41.58, 378.48, 1305.09],
    [1041.18, 1037.58, 1493.82, 778.89],
    [171.48, 217.77, 104.84, 126.6]
], dtype=np.float32)  # 银行持有贷款类资产邻接矩阵 IBA (NxM)

# 将银行-厂商矩阵转换为 Relation
IBA_rel = et_banks.dense_matrix_to_relation(IBA, dst_pool=et_firms, relation_name='IBA', attr_name='amount', threshold=0.0)

# ----------------- 高性能 Relation 操作示例 -----------------
# 构建 uid -> 位置 映射（用于高效行/列聚合）
bank_uid_to_pos = IBA_rel.build_uid_to_pos(et_banks.uid_array())
firm_uid_to_pos = IBA_rel.build_uid_to_pos(et_firms.uid_array())

# 行和 / 列和（等价于稠密矩阵的按行/按列求和）
iba_row_sum = IBA_rel.row_sum('amount', bank_uid_to_pos, n_rows=et_banks.size)
iba_col_sum = IBA_rel.col_sum('amount', firm_uid_to_pos, n_cols=et_firms.size)

# 行/列计数（度）
iba_row_degree = IBA_rel.row_count(bank_uid_to_pos, n_rows=et_banks.size)
iba_col_degree = IBA_rel.col_count(firm_uid_to_pos, n_cols=et_firms.size)

# 基本索引操作：按 src/dst uid 查询索引
bank0_uid = int(et_banks.d['i'][0])
idxs_from_bank0 = IBA_rel.indices_from_uid(bank0_uid)

# 基于属性筛选与子集抽取
mask_large = IBA_rel.get_attr('amount') > 1000
iba_large = IBA_rel.take(mask_large)

# 排序与 Top-K
order_desc = IBA_rel.argsort_by('amount', ascending=False)
iba_top3 = IBA_rel.take(order_desc[:3])

# 更新与删除（示例：将最大边权置零，并删除最后一条边）
if order_desc.size > 0:
    IBA_rel.set_attr('amount', order_desc[:1], 0.0)
if IBA_rel.size > 0:
    IBA_rel.remove_by_indices([IBA_rel.size - 1])

import time

# ----------------- 第一类二维数组（实体表）索引/查询示例（分层 API） -----------------
# 索引层：返回 idx（轻量，不拷贝列数据）
# 查询层：可返回 idx/mask 或直接返回 records（会拷贝）

# 索引层：手工构造 mask -> idxs
banks_A_exb = et_banks.get_attr('A_exb')
banks_A_IB = et_banks.get_attr('A_IB_all')
mask_rich = banks_A_exb > 3000
idxs_rich = np.nonzero(mask_rich)[0]

# 查询层：用 EntityPool.query 统一封装（return_='indices'）
idxs_combo_q = et_banks._pool.query(lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000), return_='indices', include_active_only=True)

# 查询层：直接返回 records（拷贝）
records_combo = et_banks._pool.query(lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000), return_='records', include_active_only=True)

# 说明：records_combo['name'] 等是拷贝；若你只想延迟取值，请使用 indices 再对需要的列做索引。

# ----------------- 第二类二维数组（Relation）索引/查询示例（分层 API） -----------------
# 索引层：返回边的 idx（轻量） -> 便于组合
# 查询层：可返回 idx/mask 或直接返回子 Relation（拷贝）

# 索引层：按 src_uid 找到所有边索引
bank0_uid = int(et_banks.d['i'][0])
idxs_from_bank0 = IBA_rel.indices_from_uid(bank0_uid)

# 索引层：再叠加一个边属性条件（amount > 1000）得到交集（mask 组合）
mask_amount_large = IBA_rel.get_attr('amount') > 1000
idxs_amount_large = np.nonzero(mask_amount_large)[0]
idxs_inter = np.intersect1d(idxs_from_bank0, idxs_amount_large, assume_unique=False)

# 查询层：用 Relation.query 一步完成（返回 indices）
idxs_q = IBA_rel.query(src_uid=bank0_uid, edge_pred=lambda r: r.get_attr('amount') > 1000, return_='indices')

# 查询层：直接返回子 Relation（拷贝）
sub_rel = IBA_rel.query(src_uid=bank0_uid, edge_pred=lambda r: r.get_attr('amount') > 1000, return_='relation')
