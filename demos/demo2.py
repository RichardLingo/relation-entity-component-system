"""
@File   ：demo2.py
@Desc   : 示例：演示如何使用 ECSEngine 创建实体池、初始化属性并把稠密邻接矩阵转换为 Relation。

本示例已迁移到“统一 index/query 简并语法糖”的推荐写法：
- 实体表：engine.index / engine.query
- 关系表：rel.index / rel.query
"""

import sys
from pathlib import Path

import numpy as np

try:
    from ComplexSystemLab.Projects.ECS.ecs.ecsengine import ECSEngine  # #NOTE 导入引擎模块。不能改动该行!
except ModuleNotFoundError:
    # 允许直接运行本 demo：把仓库根（ComplexSystemLab 的父目录）加入 sys.path
    # 这样在未设置 PYTHONPATH 的情况下也能运行。
    this_file = Path(__file__).resolve()
    # 结构：.../ComplexSystemLab/ComplexSystemLab/Projects/ECS/demos/demo2.py
    # 要能 import ComplexSystemLab.*，需要把“外层 ComplexSystemLab 目录”加入 sys.path
    repo_root = this_file.parents[4]  # .../ComplexSystemLab
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ComplexSystemLab.Projects.ECS.ecs.ecsengine import ECSEngine  # noqa: E402

# from ComplexSystemLab.Projects.ECS.ecs.ecsengine import dense_to_relation
# from ComplexSystemLab.Projects.ECS.ecs.ecsengine import EntityPool

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



# ----------------- 第一类二维数组（实体表）索引/查询示例（统一 index/query） -----------------
# 索引层：返回 idx（轻量，不拷贝列数据）
idxs_combo = et_banks.index(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='indices',
    include_active_only=True
)

# 查询层：返回 view（只保存 idx + 引用；需要哪列再取哪列）
banks_view = et_banks.query(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='view',
    include_active_only=True
)
banks_view_names = banks_view['name']

# 查询层：返回 records（材料化拷贝）
records_combo = et_banks.query(
    lambda p: (p.get_attr('A_exb') > 3000) & (p.get_attr('A_IB_all') > 1000),
    return_='records',
    include_active_only=True
)

# ----------------- 第二类二维数组（Relation）索引/查询示例（统一 index/query） -----------------
# 索引层：支持 src_uid/dst_uid 单 uid 或多 uid（OR），edge_pred 支持 mask/callable
bank_uids = et_banks.uid_array()
src_uids_or = [int(bank_uids[0]), int(bank_uids[1])]
edge_idxs_or = IBA_rel.index(src_uid=src_uids_or, edge_pred=(IBA_rel.get_attr('amount') > 1000), return_='indices')

# 查询层：返回 view（轻量，像表一样访问列）
edge_view = IBA_rel.query(src_uid=int(bank_uids[0]), edge_pred=(IBA_rel.get_attr('amount') > 1000), return_='view')
edge_dst_uids = edge_view['dst_uid']

# 查询层：返回子 Relation（拷贝）
sub_rel = IBA_rel.query(src_uid=int(bank_uids[0]), edge_pred=(IBA_rel.get_attr('amount') > 1000), return_='relation')

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

# 排序与 Top-K（性能示例）
order_desc = IBA_rel.argsort_by('amount', ascending=False)
iba_top3 = IBA_rel.take(order_desc[:3])

# 更新与删除（示例：将最大边权置零，并删除最后一条边）
if order_desc.size > 0:
    IBA_rel.set_attr('amount', order_desc[:1], 0.0)
if IBA_rel.size > 0:
    IBA_rel.remove_by_indices([IBA_rel.size - 1])


# ----------------- 赋值操作示例（索引后写入 / 查询后写入 / view 回写） -----------------
# 1) 实体表：按索引对单列赋值（标量广播）
et_banks.set_attr('A_exb', [0, 1], 9999.0, backend='auto')

# 2) 实体表：按同一索引批量赋值（多列）
et_banks.assign(
    [2, 3],
    {
        'A_exb': np.array([7000.0, 7100.0], dtype=np.float32),
        'A_IB_all': np.array([1500.0, 1600.0], dtype=np.float32),
    },
    backend='dense'
)

# 3) 实体表：按查询条件直接赋值（where_set）
hit_bank_idxs = et_banks.where_set(
    lambda p: p.get_attr('A_IB_all') > 2000,
    {'Z_IB_all': 0.0},
    backend='auto',
    include_active_only=True,
)

# 4) 实体表：view 可写语义（直接回写到底层）
bank_view_write = et_banks.query(lambda p: p.get_attr('A_exb') > 6000, return_='view', include_active_only=True)
if bank_view_write.size > 0:
    bank_view_write['Z_IB_all'] = np.full(bank_view_write.size, 123.0, dtype=np.float32)
    bank_view_write.assign({'A_IB_all': 321.0}, backend='sparse')


# 5) 关系表：按索引对单列赋值
if IBA_rel.size > 0:
    IBA_rel.set_attr('amount', slice(0, min(2, IBA_rel.size)), 888.0, backend='auto')

# 6) 关系表：按查询条件直接赋值（where_set）
bank_uids2 = et_banks.uid_array()
if bank_uids2.size > 0:
    hit_edge_idxs = IBA_rel.where_set(
        src_uid=int(bank_uids2[0]),
        edge_pred=(IBA_rel.get_attr('amount') > 500),
        values_by_attr={'amount': 555.0},
        backend='dense',
    )
else:
    hit_edge_idxs = np.empty(0, dtype=int)

# 7) 关系表：view 可写语义（直接回写到底层）
edge_view_write = IBA_rel.query(edge_pred=(IBA_rel.get_attr('amount') > 200), return_='view')
if edge_view_write.size > 0:
    edge_view_write['amount'] = np.full(edge_view_write.size, 222.0, dtype=np.float32)
    edge_view_write.assign({'amount': 333.0}, backend='sparse')


# 轻量输出：确认命中数量与部分结果
print("[赋值示例] 实体 where_set 命中数:", int(hit_bank_idxs.size))
print("[赋值示例] 关系 where_set 命中数:", int(hit_edge_idxs.size))
print("[赋值示例] 银行 A_exb 前3项:", et_banks.get_attr('A_exb')[:3])
print("[赋值示例] IBA amount 前5项:", IBA_rel.get_attr('amount')[:5])