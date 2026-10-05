"""
@File   : demo2.py
@Desc   : RECS 演示：RECS 与关系扩展示例。

本示例已迁移到推荐的统一 index/query 风格：
- 实体表：engine.index / engine.query
- 关系表：rel.index / rel.query

示例数据使用“银行”和“厂商”两个实体池，以及两类关系：
- 银行-银行：表示银行间拆借或债权敞口；
- 银行-厂商：表示银行持有的贷款类资产。
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
    # 结构：.../relation-entity-component-system/demos/demo2.py
    # 要能 import recs.*，需要把“仓库根目录”加入 sys.path
    repo_root = this_file.parents[1]  # .../relation-entity-component-system
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from recs import RECS  # noqa: E402

# 可在这里直接接入矩阵转关系与实体池的辅助函数。
# 当前 demo 为了直观展示 API，直接在文件中手工构造小规模数据。

# 1. 初始化银行实体池，初始容量为 5。
# attr_types_banks 定义银行实体表的业务列；RECS 还会自动维护内置 uid 列 i 和启用状态列 o。
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
# 厂商实体在关系表中作为银行贷款资产的目标端点。
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
# A_IB[row, col] 表示 row 银行对 col 银行的贷款敞口；0 表示两家银行之间没有这条边。
A_IB = np.array([
    [0, 1728.55, 0, 134.46, 322.23],
    [109.35, 0, 289.02, 0, 0],
    [730.99, 0, 0, 0, 0],
    [119.26, 115.69, 964.32, 0, 158.48],
    [0, 0, 0, 2717.39, 0]
], dtype=np.float32)  # 银行间拆借矩阵

# 将稠密的银行-银行矩阵转换为 Relation（基于 uid 的边表）。
# threshold=0.0 表示只把金额大于 0 的矩阵单元转成边；边属性名为 amount。
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
# src_uid 来自 bank_pool，dst_uid 来自 firm_pool，因此这是跨实体池关系。
bank_firm_rel = bank_pool.dense_matrix_to_relation(IBA, dst_pool=firm_pool, relation_name='IBA', attr_name='amount', threshold=0.0)



# ----------------- 第一类二维数组（实体表）索引/查询示例（统一 index/query） -----------------
# 索引层：返回 idx（轻量，不拷贝列数据）。
# 这里筛选“对厂商资产较高且银行间资产也较高”的银行。
# idxs_combo 是满足条件的银行位置索引；它不包含任何属性列拷贝。
idxs_combo = bank_pool.index(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='indices',
    include_active_only=True
)

# 查询层：返回 view（只保存 idx + 引用；需要哪列再取哪列）。
# view 适合后续还要读写多列，但又不想马上复制完整记录的场景。
# banks_view 是实体视图，内部只保存命中的索引和对 bank_pool 的引用。
banks_view = bank_pool.query(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='view',
    include_active_only=True
)
# banks_view_names 从视图中延迟读取 name 列，不会把其他列一起材料化。
banks_view_names = banks_view['name']

# 查询层：返回 records（材料化拷贝）。
# records 适合导出、打印或传给不应修改原始实体池的外部逻辑。
# records_combo 是普通记录拷贝，修改它不会回写到 bank_pool。
records_combo = bank_pool.query(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='records',
    include_active_only=True
)

# ----------------- 第二类二维数组（Relation）索引/查询示例（统一 index/query） -----------------
# 索引层：支持 src_uid/dst_uid 单 uid 或多 uid（OR），edge_pred 支持 mask/callable。
# 这里查询前两家银行中，贷款金额大于 1000 的银行-厂商边。
# bank_uids 是银行实体池内所有银行的稳定 uid，Relation 用 uid 而不是位置索引表示端点。
bank_uids = bank_pool.uid_array()
# src_uids_or 展示多源端 OR 查询：只要边的 src_uid 属于该列表就可能命中。
src_uids_or = [int(bank_uids[0]), int(bank_uids[1])]
# edge_idxs_or 是命中的边位置索引，可继续用于 set_attr、take 或删除等操作。
edge_idxs_or = bank_firm_rel.index(src_uid=src_uids_or, edge_pred=(bank_firm_rel.get_attr('amount') > 1000), return_='indices')

# 查询层：返回 view（轻量，像表一样访问列）
# edge_view 是关系视图，访问方式类似表；例如 edge_view['dst_uid'] 读取目标端 uid 列。
edge_view = bank_firm_rel.query(src_uid=int(bank_uids[0]), edge_pred=(bank_firm_rel.get_attr('amount') > 1000), return_='view')
# edge_dst_uids 是命中边指向的厂商 uid，可再映射回 firm_pool 中的厂商实体。
edge_dst_uids = edge_view['dst_uid']

# 查询层：返回子 Relation（拷贝）
# sub_rel 是一张新的 Relation 子表，适合脱离原关系表单独分析。
sub_rel = bank_firm_rel.query(src_uid=int(bank_uids[0]), edge_pred=(bank_firm_rel.get_attr('amount') > 1000), return_='relation')

# ----------------- 高性能 Relation 操作示例 -----------------
# 构建 uid -> 位置 映射（用于高效行/列聚合）。
# Relation 的端点是 uid；行列聚合需要把 uid 映射回连续位置。
# bank_uid_to_pos 把银行 uid 映射到银行池的位置，用于输出“每家银行一项”的聚合数组。
bank_uid_to_pos = bank_firm_rel.build_uid_to_pos(bank_pool.uid_array())
# firm_uid_to_pos 把厂商 uid 映射到厂商池的位置，用于输出“每个厂商一项”的聚合数组。
firm_uid_to_pos = bank_firm_rel.build_uid_to_pos(firm_pool.uid_array())

# 行和 / 列和（等价于稠密矩阵的按行/按列求和）
# iba_row_sum 是每家银行对所有厂商贷款金额的总和。
iba_row_sum = bank_firm_rel.row_sum('amount', bank_uid_to_pos, n_rows=bank_pool.size)
# iba_col_sum 是每个厂商被所有银行持有的贷款金额总和。
iba_col_sum = bank_firm_rel.col_sum('amount', firm_uid_to_pos, n_cols=firm_pool.size)

# 行/列计数（度）
# iba_row_degree 是每家银行连出的贷款边数量。
iba_row_degree = bank_firm_rel.row_count(bank_uid_to_pos, n_rows=bank_pool.size)
# iba_col_degree 是每个厂商连入的贷款边数量。
iba_col_degree = bank_firm_rel.col_count(firm_uid_to_pos, n_cols=firm_pool.size)

# 排序与 Top-K（性能示例）。
# argsort_by 返回边索引排序结果，take 根据索引抽取子关系。
# order_desc 是按 amount 从大到小排列的边索引。
order_desc = bank_firm_rel.argsort_by('amount', ascending=False)
# iba_top3 是贷款金额最大的 3 条银行-厂商边组成的子关系。
iba_top3 = bank_firm_rel.take(order_desc[:3])

# 更新与删除（示例：将最大边权置零，并删除最后一条边）
if order_desc.size > 0:
    bank_firm_rel.set_attr('amount', order_desc[:1], 0.0)
if bank_firm_rel.size > 0:
    bank_firm_rel.remove_by_indices([bank_firm_rel.size - 1])


# ----------------- 赋值操作示例（索引后写入 / 查询后写入 / view 回写） -----------------
# 1) 实体表：按索引对单列赋值（标量广播）。
# 这里把第 0、1 行银行的 A_exb 同时改为 9999.0。
bank_pool.set_attr('A_exb', [0, 1], 9999.0, backend='auto')

# 2) 实体表：按同一索引批量赋值（多列）。
# assign 适合一次改多个属性列；数组长度需要与目标索引数量一致，标量则会广播。
bank_pool.assign(
    [2, 3],
    {
        'A_exb': np.array([7000.0, 7100.0], dtype=np.float32),
        'A_IB_all': np.array([1500.0, 1600.0], dtype=np.float32),
    },
    backend='dense'
)

# 3) 实体表：按查询条件直接赋值（where_set）。
# where_set 会返回被命中的实体索引，便于记录本次批量写入影响了多少行。
# hit_bank_idxs 保存被 where_set 改写的银行位置索引。
hit_bank_idxs = bank_pool.where_set(
    lambda p: p.get_attr('A_IB_all') > 2000,
    {'Z_IB_all': 0.0},
    backend='auto',
    include_active_only=True,
)

# 4) 实体表：view 可写语义（直接回写到底层）。
# 对 view 的列赋值会修改原实体池，不是只改临时副本。
# bank_view_write 是可写实体视图，命中的银行会被后续赋值直接回写。
bank_view_write = bank_pool.query(lambda p: p.get_attr('A_exb') > 6000, return_='view', include_active_only=True)
if bank_view_write.size > 0:
    bank_view_write['Z_IB_all'] = np.full(bank_view_write.size, 123.0, dtype=np.float32)
    bank_view_write.assign({'A_IB_all': 321.0}, backend='sparse')


# 5) 关系表：按索引对单列赋值。
# Relation.set_attr 与实体池 set_attr 类似，只是目标变成了边表的属性列。
if bank_firm_rel.size > 0:
    bank_firm_rel.set_attr('amount', slice(0, min(2, bank_firm_rel.size)), 888.0, backend='auto')

# 6) 关系表：按查询条件直接赋值（where_set）。
# 这里只修改第一家银行发出的、金额大于 500 的贷款边。
# bank_uids2 重新读取 uid，避免前面对实体池写入后读者误以为 uid 会随属性变化。
bank_uids2 = bank_pool.uid_array()
if bank_uids2.size > 0:
    # hit_edge_idxs 保存被 where_set 改写的边位置索引。
    hit_edge_idxs = bank_firm_rel.where_set(
        src_uid=int(bank_uids2[0]),
        edge_pred=(bank_firm_rel.get_attr('amount') > 500),
        values_by_attr={'amount': 555.0},
        backend='dense',
    )
else:
    hit_edge_idxs = np.empty(0, dtype=int)

# 7) 关系表：view 可写语义（直接回写到底层）。
# 关系 view 同样是可写视图，适合对查询得到的边集合做批量更新。
# edge_view_write 是可写关系视图，后续对 amount 的赋值会回写到 bank_firm_rel。
edge_view_write = bank_firm_rel.query(edge_pred=(bank_firm_rel.get_attr('amount') > 200), return_='view')
if edge_view_write.size > 0:
    edge_view_write['amount'] = np.full(edge_view_write.size, 222.0, dtype=np.float32)
    edge_view_write.assign({'amount': 333.0}, backend='sparse')


# 轻量输出：确认命中数量与部分结果
print("[赋值示例] 实体 where_set 命中数:", int(hit_bank_idxs.size))
print("[赋值示例] 关系 where_set 命中数:", int(hit_edge_idxs.size))
print("[赋值示例] 银行 A_exb 前3项:", bank_pool.get_attr('A_exb')[:3])
print("[赋值示例] IBA amount 前5项:", bank_firm_rel.get_attr('amount')[:5])
