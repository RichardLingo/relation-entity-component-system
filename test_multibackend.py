#!/usr/bin/env python3
"""测试 RECS 多后端计算架构。

验证：
1. 后端注册表和自动检测
2. NumPy 后端（默认，零改动兼容）
3. PyTorch 后端（如果可用）
4. 字符串列强制 NumPy
5. to_numpy() 导出功能
"""

import numpy as np

# 多后端 RECS 引擎
from recs.backends import get_backend, list_backends
from ecs.ecsengine import RECS, EntityPool, Relation

print("=" * 60)
print("RECS 多后端计算架构测试")
print("=" * 60)

# 检查可用后端
print("\n1. 可用后端:", list_backends())

# ============================================================
# 测试 1: NumPy 后端（默认，零改动）
# ============================================================
print("\n" + "=" * 60)
print("测试 1: NumPy 后端（默认）")
print("=" * 60)

recs_np = RECS(64, {'x': float, 'y': float})
recs_np.add(3, x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
print(f"后端名称: {recs_np.backend.name}")
print(f"x 类型: {type(recs_np.d['x'])}")
print(f"x 值: {recs_np.d['x'][:recs_np.size]}")
print(f"y 值: {recs_np.d['y'][:recs_np.size]}")

# 验证是 NumPy 数组
assert isinstance(recs_np.d['x'], np.ndarray), "NumPy 后端下应该是 np.ndarray"
print("✓ NumPy 后端验证通过")

# ============================================================
# 测试 2: 字符串列强制 NumPy
# ============================================================
print("\n" + "=" * 60)
print("测试 2: 字符串列强制 NumPy")
print("=" * 60)

recs_str = RECS(64, {'name': 'U10', 'age': int})
recs_str.add(3, name=['Alice', 'Bob', 'Charlie'], age=[25, 30, 35])
print(f"后端名称: {recs_str.backend.name}")
print(f"name 类型: {type(recs_str.d['name'])}")
print(f"name 值: {recs_str.d['name'][:recs_str.size]}")
print(f"age 类型: {type(recs_str.d['age'])}")
print(f"age 值: {recs_str.d['age'][:recs_str.size]}")

# 验证字符串列是 NumPy
assert isinstance(recs_str.d['name'], np.ndarray), "字符串列应该是 np.ndarray"
assert recs_str.d['name'].dtype.kind == 'U', "字符串列应该是 Unicode 类型"
print("✓ 字符串列强制 NumPy 验证通过")

# ============================================================
# 测试 3: PyTorch 后端（如果可用）
# ============================================================
print("\n" + "=" * 60)
print("测试 3: PyTorch 后端")
print("=" * 60)

try:
    import torch
    recs_torch = RECS(64, {'x': float, 'y': float}, backend='torch')
    recs_torch.add(3, x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    print(f"后端名称: {recs_torch.backend.name}")
    print(f"x 类型: {type(recs_torch.d['x'])}")
    print(f"x 值: {recs_torch.d['x'][:recs_torch.size]}")
    print(f"x 设备: {recs_torch.d['x'].device}")
    
    # 验证是 PyTorch Tensor
    assert isinstance(recs_torch.d['x'], torch.Tensor), "PyTorch 后端下应该是 torch.Tensor"
    print("✓ PyTorch 后端验证通过")
    
    # 测试 to_numpy() 导出
    all_np = recs_torch.to_numpy()
    x_np = all_np['x']
    print(f"\n导出为 NumPy: {type(x_np)}")
    print(f"导出值: {x_np}")
    assert isinstance(x_np, np.ndarray), "to_numpy() 应该返回 np.ndarray"
    print("✓ to_numpy() 导出验证通过")
    
except ImportError as e:
    print(f"PyTorch 未安装，跳过 PyTorch 后端测试: {e}")

# ============================================================
# 测试 4: MLX 后端（如果可用）
# ============================================================
print("\n" + "=" * 60)
print("测试 4: MLX 后端")
print("=" * 60)

try:
    recs_mlx = RECS(64, {'x': float, 'y': float}, backend='mlx')
    recs_mlx.add(3, x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    print(f"后端名称: {recs_mlx.backend.name}")
    print(f"x 类型: {type(recs_mlx.d['x'])}")
    print(f"x 值: {recs_mlx.d['x'][:recs_mlx.size]}")
    
    # 验证是 MLX 数组
    import mlx.core as mx
    assert isinstance(recs_mlx.d['x'], mx.array), "MLX 后端下应该是 mx.array"
    print("✓ MLX 后端验证通过")
    
except ImportError:
    print("MLX 未安装，跳过 MLX 后端测试")

# ============================================================
# 测试 5: JAX 后端（如果可用）
# ============================================================
print("\n" + "=" * 60)
print("测试 5: JAX 后端")
print("=" * 60)

try:
    import jax.numpy as jnp
    # JAX 数组是不可变的，RECS 的 add/remove 等修改操作不支持
    # 这里只测试后端注册和只读操作
    from recs.backends import get_backend
    jax_backend = get_backend('jax')
    print(f"后端名称: {jax_backend.name}")
    print(f"设备: {jax_backend.device}")
    
    # 测试只读操作
    arr = jax_backend.zeros(10, dtype=float)
    print(f"zeros 类型: {type(arr)}")
    assert isinstance(arr, jnp.ndarray), "JAX 后端下应该是 jnp.ndarray"
    print("✓ JAX 后端验证通过（仅支持只读操作）")
    print("注意：JAX 数组不可变，RECS 的 add/remove/enable 等修改操作不支持")
    
except ImportError:
    print("JAX 未安装，跳过 JAX 后端测试")

# ============================================================
# 测试 6: TensorFlow 后端（如果可用）
# ============================================================
print("\n" + "=" * 60)
print("测试 6: TensorFlow 后端")
print("=" * 60)

try:
    import tensorflow as tf
    recs_tf = RECS(64, {'x': float, 'y': float}, backend='tensorflow')
    recs_tf.add(3, x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    print(f"后端名称: {recs_tf.backend.name}")
    print(f"x 类型: {type(recs_tf.d['x'])}")
    print(f"x 值: {recs_tf.d['x'][:recs_tf.size]}")
    
    # 验证是 TensorFlow Tensor
    assert isinstance(recs_tf.d['x'], tf.Tensor), "TensorFlow 后端下应该是 tf.Tensor"
    print("✓ TensorFlow 后端验证通过")
    
except ImportError as e:
    print(f"TensorFlow 未安装，跳过 TensorFlow 后端测试: {e}")
    
except ImportError:
    print("TensorFlow 未安装，跳过 TensorFlow 后端测试")

# ============================================================
# 测试 7: EntityPool 多后端
# ============================================================
print("\n" + "=" * 60)
print("测试 7: EntityPool 多后端")
print("=" * 60)

pool_np = EntityPool(64, {'x': float}, backend='numpy')
pool_np.add(3, x=[1.0, 2.0, 3.0])
print(f"EntityPool 后端: {pool_np.backend.name}")
print(f"x 类型: {type(pool_np.d['x'])}")
assert isinstance(pool_np.d['x'], np.ndarray)
print("✓ EntityPool NumPy 后端验证通过")

try:
    pool_torch = EntityPool(64, {'x': float}, backend='torch')
    pool_torch.add(3, x=[1.0, 2.0, 3.0])
    print(f"EntityPool 后端: {pool_torch.backend.name}")
    print(f"x 类型: {type(pool_torch.d['x'])}")
    import torch
    assert isinstance(pool_torch.d['x'], torch.Tensor)
    print("✓ EntityPool PyTorch 后端验证通过")
except ImportError:
    print("PyTorch 未安装，跳过 EntityPool PyTorch 测试")

# ============================================================
# 测试 8: Relation 多后端
# ============================================================
print("\n" + "=" * 60)
print("测试 8: Relation 多后端")
print("=" * 60)

rel_np = Relation('test', capacity=64, attr_dtypes={'weight': float}, backend='numpy')
rel_np.add([0, 1, 2], [1, 2, 0], weight=[0.5, 0.8, 0.3])
print(f"Relation 后端: {rel_np.backend.name}")
print(f"weight 类型: {type(rel_np.d['weight'])}")
assert isinstance(rel_np.d['weight'], np.ndarray)
print("✓ Relation NumPy 后端验证通过")

try:
    rel_torch = Relation('test', capacity=64, attr_dtypes={'weight': float}, backend='torch')
    rel_torch.add([0, 1, 2], [1, 2, 0], weight=[0.5, 0.8, 0.3])
    print(f"Relation 后端: {rel_torch.backend.name}")
    print(f"weight 类型: {type(rel_torch.d['weight'])}")
    import torch
    assert isinstance(rel_torch.d['weight'], torch.Tensor)
    print("✓ Relation PyTorch 后端验证通过")
except ImportError:
    print("PyTorch 未安装，跳过 Relation PyTorch 测试")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print("✓ 所有测试通过！")
print("✓ RECS 多后端计算架构工作正常")
print("✓ 默认 NumPy 后端零改动兼容")
print("✓ 字符串列强制 NumPy")
print("✓ 可选后端（PyTorch/MLX/JAX/TF）按需加载")
print("✓ to_numpy() 导出功能正常")
