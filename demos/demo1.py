"""
@File   ：demo1.py
@Desc   : 
"""

import numpy as np
from ComplexSystemLab.Projects.ECS.ecs.ecsengine import ECSEngine  # #NOTE 导入引擎模块。不能改动该行！

# 1. 初始化引擎，初始容量为4，包含自定义属性 position（float）、velocity（float）
attr_types = {'position': np.float32, 'velocity': np.float32}
et_010 = ECSEngine(capacity=4, attr_dtypes=attr_types)

# 2. 单个添加实体，赋值部分属性
idx1 = et_010.add(position=1.0, velocity=2.0)  # returns array of indices
idx2 = et_010.add(position=3.0, velocity=4.0)

# 3. 批量添加实体，赋值部分属性
idxs = et_010.add(3, position=5.0, velocity=[6.0, 7.0, 8.0])

# 4. 动态添加新属性 health，初始值为100
et_010.add_attribute('health', dtype=np.int32, default=100)

# 5. 启用/禁用实体
et_010.remove(idx1)  # 禁用第一个实体
e = et_010.get_attr('o')  # 获取所有实体的on状态
et_010.enable(idx1)  # 重新启用第一个实体

# 6. 获取所有激活实体下标，访问属性数组
active_indices = et_010.get_active_indices()
positions = et_010.get_attr('position')
healths = et_010.get_attr('health')

# 7. 展示部分结果输出
print(f"Active indices: {active_indices}")
print(f"All positions: {positions}")
print(f"All healths: {healths}")
print(f"On status: {e}")
print(f"Entity i (uid): {et_010.get_attr('i')}")
