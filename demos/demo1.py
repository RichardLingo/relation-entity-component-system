"""
@File   : demo1.py
@Desc   : RECS demo: RECS + relation extension examples.
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

# 1. 初始化 RECS 引擎，初始容量为 4，包含自定义属性 position（float）和 velocity（float）
attr_types = {'position': np.float32, 'velocity': np.float32}
recs_engine = RECS(capacity=4, attr_dtypes=attr_types)

# 2. 单个添加实体，赋值部分属性
first_idx = recs_engine.add(position=1.0, velocity=2.0)  # returns array of indices
second_idx = recs_engine.add(position=3.0, velocity=4.0)

# 3. 批量添加实体，赋值部分属性
batch_idxs = recs_engine.add(3, position=5.0, velocity=[6.0, 7.0, 8.0])

# 4. 动态添加新属性 health，初始值为 100
recs_engine.add_attribute('health', dtype=np.int32, default=100)

# 5. 启用/禁用实体
recs_engine.remove(first_idx)  # 禁用第一个实体
active_mask = recs_engine.get_attr('o')  # 获取所有实体的 on 状态
recs_engine.enable(first_idx)  # 重新启用第一个实体

# 6. 获取所有激活实体下标，访问属性数组
active_indices = recs_engine.get_active_indices()
positions = recs_engine.get_attr('position')
healths = recs_engine.get_attr('health')

# 7. 展示部分结果输出
print(f"Active indices: {active_indices}")
print(f"All positions: {positions}")
print(f"All healths: {healths}")
print(f"On status: {active_mask}")
print(f"Entity i (uid): {recs_engine.get_attr('i')}")

