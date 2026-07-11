"""
@File   : demo_multi_backend.py
@Desc   : RECS 多后端计算演示

展示 RECS 框架在不同计算后端（NumPy、PyTorch、MLX、JAX、TensorFlow）上的使用方式。
演示如何：
1. 在不同后端之间切换
2. 使用统一 API 进行实体池操作
3. 处理后端特定的数据类型转换
4. 利用后端特性进行高性能计算
"""

import sys
from pathlib import Path

import numpy as np

# 确保可以导入 recs 模块
this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from recs import RECS


def check_backend_available(backend_name):
    """检查后端是否可用"""
    try:
        if backend_name == 'torch':
            import torch
            return True
        elif backend_name == 'mlx':
            import mlx.core as mx
            return True
        elif backend_name == 'jax':
            import jax.numpy as jnp
            return True
        elif backend_name == 'tensorflow':
            import tensorflow as tf
            return True
        elif backend_name == 'numpy':
            return True
    except ImportError:
        return False
    return False


def demo_numpy_backend():
    """演示 NumPy 后端"""
    print("\n" + "="*60)
    print("NumPy 后端演示")
    print("="*60)
    
    # 创建实体池
    attr_types = {
        'name': 'U10',
        'position': np.float32,
        'velocity': np.float32,
        'health': np.int32,
    }
    
    pool = RECS(capacity=10, attr_dtypes=attr_types, backend='numpy')
    
    # 添加实体
    pool.add(5, 
             name=['Entity1', 'Entity2', 'Entity3', 'Entity4', 'Entity5'],
             position=[1.0, 2.0, 3.0, 4.0, 5.0],
             velocity=[0.1, 0.2, 0.3, 0.4, 0.5],
             health=[100, 90, 80, 70, 60])
    
    print(f"创建实体数: {pool.size}")
    print(f"位置数据: {pool.get_attr('position')}")
    print(f"健康值: {pool.get_attr('health')}")
    
    # 查询操作
    high_health = pool.query(
        lambda p: p.get_attr('health') > 75,
        return_='indices',
        include_active_only=True
    )
    print(f"健康值 > 75 的实体索引: {high_health}")
    
    # 条件赋值
    pool.where_set(
        lambda p: p.get_attr('health') < 80,
        {'health': 80}
    )
    print(f"条件赋值后健康值: {pool.get_attr('health')}")
    
    # 批量更新
    pool.assign([0, 1, 2], {'position': 10.0, 'velocity': 1.0})
    print(f"批量更新后位置: {pool.get_attr('position')}")
    
    print("✓ NumPy 后端演示完成")


def demo_torch_backend():
    """演示 PyTorch 后端"""
    if not check_backend_available('torch'):
        print("\n⚠ PyTorch 不可用，跳过演示")
        return
    
    print("\n" + "="*60)
    print("PyTorch 后端演示")
    print("="*60)
    
    import torch
    
    # 创建实体池
    attr_types = {
        'name': 'U10',
        'position': torch.float32,
        'velocity': torch.float32,
        'health': torch.int64,  # PyTorch 默认整数类型
    }
    
    # 尝试使用 MPS（Apple Silicon）或 CUDA
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    pool = RECS(capacity=10, attr_dtypes=attr_types, backend='torch')
    
    # 添加实体
    pool.add(5,
             name=['Entity1', 'Entity2', 'Entity3', 'Entity4', 'Entity5'],
             position=[1.0, 2.0, 3.0, 4.0, 5.0],
             velocity=[0.1, 0.2, 0.3, 0.4, 0.5],
             health=[100, 90, 80, 70, 60])
    
    print(f"创建实体数: {pool.size}")
    print(f"位置数据类型: {type(pool.get_attr('position'))}")
    print(f"位置数据: {pool.get_attr('position')}")
    
    # 查询操作
    high_health = pool.query(
        lambda p: p.get_attr('health') > 75,
        return_='indices',
        include_active_only=True
    )
    print(f"健康值 > 75 的实体索引: {high_health}")
    
    # 条件赋值
    pool.where_set(
        lambda p: p.get_attr('health') < 80,
        {'health': 80}
    )
    print(f"条件赋值后健康值: {pool.get_attr('health')}")
    
    # 批量更新
    pool.assign([0, 1, 2], {'position': 10.0, 'velocity': 1.0})
    print(f"批量更新后位置: {pool.get_attr('position')}")
    
    print("✓ PyTorch 后端演示完成")


def demo_mlx_backend():
    """演示 MLX 后端（Apple Silicon）"""
    if not check_backend_available('mlx'):
        print("\n⚠ MLX 不可用，跳过演示")
        return
    
    print("\n" + "="*60)
    print("MLX 后端演示")
    print("="*60)
    
    import mlx.core as mx
    
    # 创建实体池
    attr_types = {
        'name': 'U10',
        'position': mx.float32,
        'velocity': mx.float32,
        'health': mx.int32,
    }
    
    pool = RECS(capacity=10, attr_dtypes=attr_types, backend='mlx')
    
    # 添加实体
    pool.add(5,
             name=['Entity1', 'Entity2', 'Entity3', 'Entity4', 'Entity5'],
             position=[1.0, 2.0, 3.0, 4.0, 5.0],
             velocity=[0.1, 0.2, 0.3, 0.4, 0.5],
             health=[100, 90, 80, 70, 60])
    
    print(f"创建实体数: {pool.size}")
    print(f"位置数据类型: {type(pool.get_attr('position'))}")
    print(f"位置数据: {pool.get_attr('position')}")
    
    # 查询操作
    high_health = pool.query(
        lambda p: p.get_attr('health') > 75,
        return_='indices',
        include_active_only=True
    )
    print(f"健康值 > 75 的实体索引: {high_health}")
    
    # 条件赋值
    pool.where_set(
        lambda p: p.get_attr('health') < 80,
        {'health': 80}
    )
    print(f"条件赋值后健康值: {pool.get_attr('health')}")
    
    # 批量更新
    pool.assign([0, 1, 2], {'position': 10.0, 'velocity': 1.0})
    print(f"批量更新后位置: {pool.get_attr('position')}")
    
    print("✓ MLX 后端演示完成")


def demo_jax_backend():
    """演示 JAX 后端"""
    if not check_backend_available('jax'):
        print("\n⚠ JAX 不可用，跳过演示")
        return
    
    print("\n" + "="*60)
    print("JAX 后端演示")
    print("="*60)
    
    # 启用 JAX 64 位模式以避免 int64 截断警告
    import jax
    jax.config.update("jax_enable_x64", True)
    
    import jax.numpy as jnp
    
    # 创建实体池
    attr_types = {
        'name': 'U10',
        'position': jnp.float32,
        'velocity': jnp.float32,
        'health': jnp.int32,
    }
    
    pool = RECS(capacity=10, attr_dtypes=attr_types, backend='jax')
    
    # 添加实体
    pool.add(5,
             name=['Entity1', 'Entity2', 'Entity3', 'Entity4', 'Entity5'],
             position=[1.0, 2.0, 3.0, 4.0, 5.0],
             velocity=[0.1, 0.2, 0.3, 0.4, 0.5],
             health=[100, 90, 80, 70, 60])
    
    print(f"创建实体数: {pool.size}")
    print(f"位置数据类型: {type(pool.get_attr('position'))}")
    print(f"位置数据: {pool.get_attr('position')}")
    
    # 查询操作
    high_health = pool.query(
        lambda p: p.get_attr('health') > 75,
        return_='indices',
        include_active_only=True
    )
    print(f"健康值 > 75 的实体索引: {high_health}")
    
    # 条件赋值
    pool.where_set(
        lambda p: p.get_attr('health') < 80,
        {'health': 80}
    )
    print(f"条件赋值后健康值: {pool.get_attr('health')}")
    
    # 批量更新
    pool.assign([0, 1, 2], {'position': 10.0, 'velocity': 1.0})
    print(f"批量更新后位置: {pool.get_attr('position')}")
    
    print("✓ JAX 后端演示完成")


def demo_backend_comparison():
    """比较不同后端的性能"""
    print("\n" + "="*60)
    print("后端性能比较（大规模测试）")
    print("="*60)
    
    import time
    
    # 启用 JAX 64 位模式
    if check_backend_available('jax'):
        import jax
        jax.config.update("jax_enable_x64", True)
    
    backends = ['numpy']
    if check_backend_available('torch'):
        backends.append('torch')
    if check_backend_available('mlx'):
        backends.append('mlx')
    if check_backend_available('jax'):
        backends.append('jax')
    
    # 增大测试规模以显示差异
    n_entities = 500000  # 50 万实体
    n_iterations = 50    # 50 次迭代
    
    for backend in backends:
        try:
            attr_types = {
                'position': np.float32,
                'velocity': np.float32,
            }
            
            pool = RECS(capacity=n_entities, attr_dtypes=attr_types, backend=backend)
            pool.add(n_entities,
                    position=np.random.randn(n_entities).astype(np.float32),
                    velocity=np.random.randn(n_entities).astype(np.float32))
            
            start = time.time()
            for _ in range(n_iterations):
                # 模拟物理更新：position += velocity * dt
                positions = pool.get_attr('position')
                velocities = pool.get_attr('velocity')
                dt = 0.016  # 60 FPS
                
                # 使用后端特定的操作
                if backend == 'numpy':
                    new_positions = positions + velocities * dt
                elif backend == 'torch':
                    import torch
                    new_positions = positions + velocities * dt
                    # PyTorch 需要转换为 numpy 数组
                    new_positions = new_positions.cpu().numpy()
                elif backend == 'mlx':
                    import mlx.core as mx
                    new_positions = positions + velocities * dt
                elif backend == 'jax':
                    import jax.numpy as jnp
                    new_positions = positions + velocities * dt
                
                pool.assign(list(range(n_entities)), {'position': new_positions})
            
            elapsed = time.time() - start
            print(f"{backend:12s}: {elapsed:.3f}s ({n_iterations} iterations, {n_entities} entities)")
            
        except Exception as e:
            print(f"{backend:12s}: 错误 - {e}")


def main():
    """主函数"""
    print("RECS 多后端计算演示")
    print("="*60)
    
    # 运行各个后端演示
    demo_numpy_backend()
    demo_torch_backend()
    demo_mlx_backend()
    demo_jax_backend()
    
    # 性能比较
    demo_backend_comparison()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)


if __name__ == '__main__':
    main()
