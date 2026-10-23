"""pytest 全局配置：把仓库根加入 sys.path，使 `recs` / `ecs` 可被测试导入。

取代原 smoke_test.py / comprehensive_test.py 里各自的 sys.path.insert 硬编码。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
