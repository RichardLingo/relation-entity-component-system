"""
@File   : demo1.py
@Desc   : RECS 演示：RECS 与关系扩展示例。

本文件演示 RECS 最基础的实体池用法：创建属性列、添加实体、动态增加属性、
启用/禁用实体，以及读取属性数组。它适合作为第一次阅读 RECS API 的入口。
"""

import sys
from pathlib import Path
import numpy as np

# 允许直接运行本 demo：把仓库根目录加入 sys.path
this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]  # .../relation-entity-component-system
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from recs import RECS  # #NOTE 导入 RECS 主入口。不能改动该行!

# 1. 初始化 RECS 引擎。
# capacity 是预分配容量；attr_dtypes 定义业务属性列。RECS 会额外维护内置列：
# - i：实体 uid，用于在关系表中稳定引用实体；
# - o：实体启用状态，False 的实体会被“仅激活实体”查询过滤。
# attr_types 是“列名 -> 数据类型”的声明，决定每个实体可存储哪些业务属性。
attr_types = {'position': np.float32, 'velocity': np.float32}
# recs_engine 是实体池实例；后续所有实体增删、属性访问都通过它完成。
recs_engine = RECS(capacity=4, attr_dtypes=attr_types)

# 2. 单个添加实体，赋值部分属性。
# add 不传数量时添加一个实体；返回值是实体在底层数组中的索引数组。
first_idx = recs_engine.add(position=1.0, velocity=2.0)  # 返回索引数组
# second_idx 保存第二个实体的位置索引，后续可用它定点读取或写入该实体。
second_idx = recs_engine.add(position=3.0, velocity=4.0)

# 3. 批量添加实体，赋值部分属性。
# 第一个位置参数 3 表示一次添加 3 个实体；标量 position 会广播到所有新增实体，
# 列表 velocity 会按新增实体顺序逐项写入。
# batch_idxs 保存本次批量添加得到的 3 个实体位置索引。
batch_idxs = recs_engine.add(3, position=5.0, velocity=[6.0, 7.0, 8.0])

# 4. 动态添加新属性 health，初始值为 100。
# 已存在实体会收到默认值；之后新增实体也会拥有这列属性。
recs_engine.add_attribute('health', dtype=np.int32, default=100)

# 5. 启用/禁用实体。
# remove 在这个演示中等价于把实体标记为未启用，而不是压缩数组或改变其他实体索引。
recs_engine.remove(first_idx)  # 禁用第一个实体
active_mask = recs_engine.get_attr('o')  # 获取所有实体的 on 状态
recs_engine.enable(first_idx)  # 重新启用第一个实体

# 6. 获取所有激活实体下标，访问属性数组。
# get_attr 返回底层属性列数组，适合直接交给 NumPy 或后端数组操作。
# active_indices 是当前启用实体的位置索引；禁用实体不会出现在这个数组中。
active_indices = recs_engine.get_active_indices()
# positions 和 healths 是完整属性列数组，可用 active_indices 再筛选激活实体。
positions = recs_engine.get_attr('position')
healths = recs_engine.get_attr('health')

# 7. 展示部分结果输出
print(f"Active indices: {active_indices}")
print(f"All positions: {positions}")
print(f"All healths: {healths}")
print(f"On status: {active_mask}")
print(f"Entity i (uid): {recs_engine.get_attr('i')}")
