"""
@File   : demo3.py
@Desc   : RECS 演示：RECS 与关系扩展示例。

本文件保留较早的分层 API 写法，用于对比 demo2 中推荐的统一 index/query 风格。
重点是展示如何手工构造 mask、idx，并把这些索引用于 Relation 查询和子集抽取。
"""

import sys
from pathlib import Path

import numpy as np

try:
    from recs import RECS  # #NOTE 导入 RECS 主入口。不能改动该行!
except ModuleNotFoundError:
    # 允许直接运行本 demo：把仓库根目录加入 sys.path
    # 这样在未设置 PYTHONPATH 的情况下也能运行。
    this_file = Path(__file__).resolve()
    # 结构：.../relation-entity-component-system/demos/demo3.py
    # 要能 import recs.*，需要把“仓库根目录”加入 sys.path
    repo_root = this_file.parents[1]  # .../relation-entity-component-system
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from recs import RECS  # noqa: E402

# 可在这里直接接入矩阵转关系与实体池的辅助函数。
# 当前 demo 直接内嵌小规模矩阵，方便读者逐行理解实体池和关系表之间的映射。

# 1. 初始化银行实体池，初始容量为 5。
# 每个属性列都是一个数组，数组下标就是实体在池内的位置；内置 uid 列 i 用于跨关系表引用。
attr_types_banks = {
    'name': 'U10',  # 银行名称，字符串类型，最大长度10
    'A_exb': np.float32,  # 对厂商的资产
    'A_IB_all': np.float32,  # 银间的资产
    'Z_IB_all': np.float32  # 银间的负债
}
bank_pool = RECS(capacity=5, attr_dtypes=attr_types_banks)
bank_pool.d['o'][:] = True  # 启用所有银行实体
bank_pool.d['name'][:] = ['Bank 1', 'Bank 2', 'Bank 3', 'Bank 4', 'Bank 5']  # 初始化银行名称
bank_pool.d['A_exb'][:] = [1631.73, 4303.24, 3379.72, 4351.47, 620.69]  # 初始化对厂商的资产
bank_pool.d['A_IB_all'][:] = [2185.24, 398.37, 730.99, 1357.75, 2717.39]  # 初始化银行间的资产
bank_pool.d['Z_IB_all'][:] = [959.6, 1844.24, 1253.34, 2851.85, 480.71]  # 初始化银行间的负债
# 设置逻辑 size，使 pool 知道已有实体数量
bank_pool._pool.size = 5
bank_pool._pool.d['o'][:bank_pool._pool.size] = True

# 2. 初始化厂商实体池，初始容量为 4。
# 银行-厂商关系表的目标端点会引用这些厂商实体的 uid。
attr_types_firms = {
    'name': 'U10',  # 厂商名称，字符串类型，最大长度10
    'production_rate': np.float32,  # 生产率
    'DA': np.float32  # 贷款金额
}
firm_pool = RECS(capacity=4, attr_dtypes=attr_types_firms)
firm_pool.d['o'][:] = True  # 启用所有厂商实体
firm_pool.d['name'][:] = ['Firm 1', 'Firm 2', 'Firm 3', 'Firm 4']  # 初始化厂商名称
firm_pool.d['production_rate'][:] = [1.2, 0.9, 1.1, 1.3]  # 初始化生产率
firm_pool.d['DA'][:] = [4361.83, 2575.97, 4259.21, 3089.84]  # 初始化贷款金额
# 设置逻辑 size
firm_pool._pool.size = 4
firm_pool._pool.d['o'][:firm_pool._pool.size] = True

pass  # 调试占位

# 3. 银行间拆借矩阵。
# A_IB[row, col] 表示 row 银行对 col 银行的债权敞口。
A_IB = np.array([
    [0, 1728.55, 0, 134.46, 322.23],
    [109.35, 0, 289.02, 0, 0],
    [730.99, 0, 0, 0, 0],
    [119.26, 115.69, 964.32, 0, 158.48],
    [0, 0, 0, 2717.39, 0]
], dtype=np.float32)  # 银行间拆借矩阵

# 将稠密的银行-银行矩阵转换为 Relation（基于 uid 的边表）。
# 非零矩阵单元会变成边，金额写入 amount 属性列。
bank_bank_rel = bank_pool.dense_matrix_to_relation(A_IB, dst_pool=bank_pool, relation_name='A_IB', attr_name='amount', threshold=0.0)

# 4. 银行-厂商贷款矩阵。
# IBA[row, col] 表示 row 银行持有 col 厂商相关贷款资产的金额。
IBA = np.array([
    [685.08, 115.72, 229.3, 601.64],
    [809.53, 1163.32, 2052.77, 277.62],
    [1654.57, 41.58, 378.48, 1305.09],
    [1041.18, 1037.58, 1493.82, 778.89],
    [171.48, 217.77, 104.84, 126.6]
], dtype=np.float32)  # 银行持有贷款类资产邻接矩阵 IBA (NxM)

# 将银行-厂商矩阵转换为 Relation。
# 这张关系表跨越两个实体池：源端是银行，目标端是厂商。
bank_firm_rel = bank_pool.dense_matrix_to_relation(IBA, dst_pool=firm_pool, relation_name='IBA', attr_name='amount', threshold=0.0)

# ----------------- 高性能 Relation 操作示例 -----------------
# 构建 uid -> 位置 映射（用于高效行/列聚合）。
# row_sum/col_sum 需要知道每个 uid 对应输出数组的哪个位置。
# bank_uid_to_pos 把银行 uid 映射成银行池位置，保证聚合结果可以按银行顺序对齐。
bank_uid_to_pos = bank_firm_rel.build_uid_to_pos(bank_pool.uid_array())
# firm_uid_to_pos 把厂商 uid 映射成厂商池位置，保证聚合结果可以按厂商顺序对齐。
firm_uid_to_pos = bank_firm_rel.build_uid_to_pos(firm_pool.uid_array())

# 行和 / 列和（等价于稠密矩阵的按行/按列求和）
# iba_row_sum 是每家银行贷款资产金额的行聚合结果。
iba_row_sum = bank_firm_rel.row_sum('amount', bank_uid_to_pos, n_rows=bank_pool.size)
# iba_col_sum 是每个厂商被银行持有贷款金额的列聚合结果。
iba_col_sum = bank_firm_rel.col_sum('amount', firm_uid_to_pos, n_cols=firm_pool.size)

# 行/列计数（度）
# iba_row_degree 是每家银行发出的贷款边数量。
iba_row_degree = bank_firm_rel.row_count(bank_uid_to_pos, n_rows=bank_pool.size)
# iba_col_degree 是每个厂商接收的贷款边数量。
iba_col_degree = bank_firm_rel.col_count(firm_uid_to_pos, n_cols=firm_pool.size)

# 基本索引操作：按 src/dst uid 查询索引。
# indices_from_uid 返回从指定银行发出的所有边索引。
# bank0_uid 是第 0 家银行的稳定 uid，用于演示按源端查询关系。
bank0_uid = int(bank_pool.d['i'][0])
# idxs_from_bank0 是所有从第 0 家银行发出的边位置索引。
idxs_from_bank0 = bank_firm_rel.indices_from_uid(bank0_uid)

# 基于属性筛选与子集抽取。
# mask_large 是布尔掩码；take 会把命中的边复制成一个子 Relation。
# mask_large 标记 amount 大于 1000 的边。
mask_large = bank_firm_rel.get_attr('amount') > 1000
# iba_large 是所有大额贷款边组成的子关系。
iba_large = bank_firm_rel.take(mask_large)

# 排序与 Top-K。
# 先按 amount 降序得到边索引，再取前 3 条作为最大贷款敞口示例。
# order_desc 是按贷款金额降序排列的边索引。
order_desc = bank_firm_rel.argsort_by('amount', ascending=False)
# iba_top3 是金额最大的 3 条贷款边组成的子关系。
iba_top3 = bank_firm_rel.take(order_desc[:3])

# 更新与删除（示例：将最大边权置零，并删除最后一条边）
if order_desc.size > 0:
    bank_firm_rel.set_attr('amount', order_desc[:1], 0.0)
if bank_firm_rel.size > 0:
    bank_firm_rel.remove_by_indices([bank_firm_rel.size - 1])

import time

# ----------------- 第一类二维数组（实体表）索引/查询示例（分层 API） -----------------
# 索引层：返回 idx（轻量，不拷贝列数据）
# 查询层：可返回 idx/mask 或直接返回 records（会拷贝）

# 索引层：手工构造 mask -> idxs。
# 这是最接近 NumPy 的写法：先取列数组，再构造布尔条件，最后用 nonzero 得到行索引。
# banks_A_exb 是银行对厂商资产列的完整数组。
banks_A_exb = bank_pool.get_attr('A_exb')
# banks_A_IB 是银行间资产列的完整数组。
banks_A_IB = bank_pool.get_attr('A_IB_all')
# mask_rich 标记对厂商资产超过 3000 的银行。
mask_rich = banks_A_exb > 3000
# idxs_rich 是 mask_rich 中 True 对应的银行位置索引。
idxs_rich = np.nonzero(mask_rich)[0]

# 查询层：用 EntityPool.query 统一封装（return_='indices'）。
# 与手工 mask 相比，query 把条件函数和 active-only 过滤封装在同一个接口中。
# idxs_combo_q 是同时满足 A_exb 和 A_IB_all 条件的银行位置索引。
idxs_combo_q = bank_pool._pool.query(lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000), return_='indices', include_active_only=True)

# 查询层：直接返回 records（拷贝）
# records_combo 是查询结果的材料化拷贝，适合导出或打印。
records_combo = bank_pool._pool.query(lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000), return_='records', include_active_only=True)

# 说明：records_combo['name'] 等是拷贝；若你只想延迟取值，请使用 indices 再对需要的列做索引。

# ----------------- 第二类二维数组（Relation）索引/查询示例（分层 API） -----------------
# 索引层：返回边的 idx（轻量） -> 便于组合
# 查询层：可返回 idx/mask 或直接返回子 Relation（拷贝）

# 索引层：按 src_uid 找到所有边索引。
# bank0_uid 是稳定实体标识，不会因为实体池数组位置变化而失效。
bank0_uid = int(bank_pool.d['i'][0])
# idxs_from_bank0 再次作为分层 API 示例的基础边集合。
idxs_from_bank0 = bank_firm_rel.indices_from_uid(bank0_uid)

# 索引层：再叠加一个边属性条件（amount > 1000）得到交集（mask 组合）
# mask_amount_large 标记大额贷款边。
mask_amount_large = bank_firm_rel.get_attr('amount') > 1000
# idxs_amount_large 是所有大额贷款边的位置索引。
idxs_amount_large = np.nonzero(mask_amount_large)[0]
# idxs_inter 是“第 0 家银行发出”且“金额大于 1000”的边索引交集。
idxs_inter = np.intersect1d(idxs_from_bank0, idxs_amount_large, assume_unique=False)

# 查询层：用 Relation.query 一步完成（返回 indices）。
# 这里把 src_uid 过滤和 amount 条件合并到一个调用里。
# idxs_q 是 Relation.query 一步得到的边索引，语义与 idxs_inter 接近。
idxs_q = bank_firm_rel.query(src_uid=bank0_uid, edge_pred=lambda r: r.get_attr('amount') > 1000, return_='indices')

# 查询层：直接返回子 Relation（拷贝）
# sub_rel 是查询得到的子关系，修改它不会影响原始 bank_firm_rel。
sub_rel = bank_firm_rel.query(src_uid=bank0_uid, edge_pred=lambda r: r.get_attr('amount') > 1000, return_='relation')
