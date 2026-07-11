#!/usr/bin/env python3
"""RECS 多后端计算架构——综合边缘测试套件。

覆盖：
1. 所有后端的基本 CRUD 操作
2. 边缘情况（空池、单元素、大容量）
3. 字符串列隔离
4. Relation 操作
5. 跨后端 to_numpy 导出一致性
6. 各后端的设备信息
"""

import sys, os, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from recs.backends import get_backend, list_backends

# ============================================================
# Test Harness
# ============================================================
passed = 0
failed = 0
skipped = 0
errors = []

def test(name, fn):
    global passed, failed, skipped
    try:
        fn()
        passed += 1
        print(f"  ✓ {name}")
    except ImportError as e:
        skipped += 1
        print(f"  ⤷ {name}: 跳过 ({e})")
    except Exception as e:
        failed += 1
        errors.append((name, str(e), traceback.format_exc()))
        print(f"  ✗ {name}: {e}")

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: expected {b!r}, got {a!r}")

def _to_numpy(arr):
    """Convert any array (including backend arrays) to NumPy."""
    if isinstance(arr, np.ndarray):
        return arr
    # Check for torch tensors
    if hasattr(arr, 'cpu') and hasattr(arr, 'detach'):
        return arr.cpu().detach().numpy()
    # Default: try np.asarray
    return np.asarray(arr)

def assert_arr_eq(a, b, msg=""):
    """Compare arrays across backends."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if a_np.dtype.kind in ('U', 'S') or b_np.dtype.kind in ('U', 'S'):
        if not np.array_equal(a_np, b_np):
            raise AssertionError(f"{msg}: arrays differ\n  expected: {b_np}\n  got: {a_np}")
    else:
        if not np.allclose(a_np, b_np):
            raise AssertionError(f"{msg}: arrays differ\n  expected: {b_np}\n  got: {a_np}")

# ============================================================
# Tests
# ============================================================
available = list_backends()
print(f"可用后端: {available}")
print()

# ---- 1. 后端注册表 ----
print("=== 1. 后端注册表 ===")
def test_backend_registry():
    assert 'numpy' in available, "NumPy 必须是可用后端"
    b = get_backend('numpy')
    assert b.name == 'numpy'
test("后端注册表包含 numpy", test_backend_registry)

print()

# ---- 2. 各后端基础操作 ----
print("=== 2. 各后端基础操作 ===")

for bk in available:
    if bk == 'jax':
        continue  # JAX 数组不可变，跳过写操作测试
    def make_test(bk=bk):
        def _test():
            from ecs.engine import EntityPool, RECS
            # RECS 级别
            r = RECS(16, {'x': float, 'y': float}, backend=bk)
            r.add(5, x=[1.0, 2.0, 3.0, 4.0, 5.0], y=[10.0, 20.0, 30.0, 40.0, 50.0])
            assert r.size == 5
            x_vals = r.get_attr('x')[:5]
            assert_arr_eq(x_vals, [1.0, 2.0, 3.0, 4.0, 5.0], f"{bk} RECS.add")

            # where_set
            r.where_set(lambda p: p.get_attr('x') > 3, {'x': 99.0})
            x_vals = r.get_attr('x')[:5]
            assert_arr_eq(x_vals, [1.0, 2.0, 3.0, 99.0, 99.0], f"{bk} where_set")

            # query
            idxs = r.query(lambda p: p.get_attr('x') == 99.0, return_='indices')
            assert len(idxs) == 2, f"{bk} query indices: expected 2, got {len(idxs)}"

            # view
            view = r.query(lambda p: p.get_attr('x') == 99.0, return_='view')
            assert len(view) == 2
            view['y'] = 888.0
            y_vals = r.get_attr('y')[:5]
            assert_arr_eq(y_vals, [10.0, 20.0, 30.0, 888.0, 888.0], f"{bk} view set")

            # remove = disable (size 不变)
            r.remove(idxs)
            # 验证标记为 inactive
            for i in idxs:
                assert not r.is_active(i), f"{bk} remove: entity {i} should be inactive"

            # to_numpy
            out = r.to_numpy()
            assert isinstance(out['x'], np.ndarray)

            # enable/disable
            r.disable([0])
            assert not r.is_active(0)
            r.enable([0])
            assert r.is_active(0)

            # EntityPool 级别
            p = EntityPool(16, {'a': int}, backend=bk)
            p.add(3, a=[10, 20, 30])
            assert p.size == 3
            assert_arr_eq(p.get_attr('a')[:3], [10, 20, 30], f"{bk} EntityPool")
        return _test
    test(f"{bk} 后端完整 CRUD", make_test())

print()

# ---- 3. 字符串列隔离 ----
print("=== 3. 字符串列隔离 ===")

for bk in available:
    if bk == 'jax':
        continue  # JAX 数组不可变，跳过写操作测试
    def make_test(bk=bk):
        def _test():
            from ecs.engine import RECS
            r = RECS(16, {'name': 'U10', 'tag': 'U5', 'val': float}, backend=bk)
            r.add(4, name=['A', 'B', 'C', 'D'], tag=['x', 'y', 'z', 'w'], val=[1.0, 2.0, 3.0, 4.0])
            assert r.size == 4
            # 字符串列是 NumPy
            assert isinstance(r.d['name'], np.ndarray)
            assert isinstance(r.d['tag'], np.ndarray)
            assert r.d['name'].dtype.kind == 'U'
            # 数值列是后端类型
            if bk == 'torch':
                import torch
                assert isinstance(r.d['val'], torch.Tensor)
            # where_set 不破坏字符串
            r.where_set(lambda p: p.get_attr('val') > 2, {'val': 99.0})
            names = r.get_attr('name')[:4]
            assert_arr_eq(names, ['A', 'B', 'C', 'D'], f"{bk} string unchanged")
        return _test
    test(f"{bk} 字符串列隔离", make_test())

print()

# ---- 4. 边缘情况 ----
print("=== 4. 边缘情况 ===")

def test_empty_pool():
    from ecs.engine import RECS
    r = RECS(16, {'x': float})
    assert r.size == 0
    idxs = r.query(return_='indices')
    assert idxs is not None
    assert len(idxs) == 0
test("空池操作", test_empty_pool)

def test_single_element():
    from ecs.engine import RECS
    r = RECS(16, {'x': float})
    r.add(1, x=[42.0])
    assert r.size == 1
    assert r.get_attr('x')[0] == 42.0
    r.where_set(lambda p: p.get_attr('x') == 42.0, {'x': 0.0})
    assert r.get_attr('x')[0] == 0.0
test("单元素池", test_single_element)

def test_large_capacity():
    from ecs.engine import RECS
    r = RECS(1024, {'x': float})
    r.add(1000, x=np.random.randn(1000).tolist())
    assert r.size == 1000
    assert len(r.d['x']) >= 1000
test("大容量池 (1000 实体)", test_large_capacity)

def test_bool_mask_query():
    from ecs.engine import RECS
    r = RECS(16, {'x': float})
    r.add(5, x=[1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, False, True, False, True])
    idxs = r.query(mask, return_='indices')
    assert len(idxs) == 3
test("布尔掩码查询", test_bool_mask_query)

def test_remove_disable():
    """remove = disable，size 不变但标记为 inactive"""
    from ecs.engine import RECS
    r = RECS(16, {'x': float})
    r.add(3, x=[1.0, 2.0, 3.0])
    r.remove([0, 1, 2])
    assert r.size == 3  # remove 不改变 size
    assert not r.is_active(0)
    assert not r.is_active(1)
    assert not r.is_active(2)
test("remove=disable 测试", test_remove_disable)

print()

# ---- 5. Relation 操作 ----
print("=== 5. Relation 操作 ===")

for bk in available:
    def make_test(bk=bk):
        def _test():
            from ecs.engine import Relation
            rel = Relation('test', capacity=32, attr_dtypes={'weight': float, 'label': 'U5'}, backend=bk)
            rel.add([0, 1, 2], [1, 2, 0], weight=[0.5, 0.8, 0.3], label=['a', 'b', 'c'])
            assert rel.size == 3
            w = rel.get_attr('weight')[:3]
            assert_arr_eq(w, [0.5, 0.8, 0.3], f"{bk} Relation.add")

            # 字符串列
            lbl = rel.get_attr('label')[:3]
            assert isinstance(lbl, np.ndarray)

            # neighbors
            dsts, _ = rel.neighbors_from_uid(0)
            assert len(dsts) == 1

            # remove by indices
            rel.remove_by_indices([1])
            assert rel.size == 2

            # argsort
            order = rel.argsort_by('weight')
            assert len(order) == rel.size
        return _test
    test(f"{bk} Relation 综合操作", make_test())

print()

# ---- 6. to_numpy 导出一致性 ----
print("=== 6. to_numpy 导出一致性 ===")

for bk in available:
    if bk == 'jax':
        continue  # JAX 数组不可变，跳过
    def make_test(bk=bk):
        def _test():
            from ecs.engine import RECS
            np.random.seed(42)
            data = np.random.randn(10)
            r = RECS(32, {'x': float}, backend=bk)
            r.add(10, x=data.tolist())
            out = r.to_numpy()
            assert isinstance(out['x'], np.ndarray)
            assert np.allclose(out['x'], data), f"{bk} to_numpy 数值不一致"
        return _test
    test(f"{bk} to_numpy 一致性", make_test())

# 跨后端 to_numpy 一致性
def test_cross_backend_consistency():
    from ecs.engine import RECS
    np.random.seed(42)
    data = np.random.randn(10)
    ref = None
    for bk in available:
        if bk == 'jax':
            continue  # JAX 只读，跳过
        r = RECS(32, {'x': float}, backend=bk)
        r.add(10, x=data.tolist())
        out = r.to_numpy()
        if ref is None:
            ref = out['x'].copy()
        else:
            assert np.allclose(ref, out['x']), f"{bk} 与 NumPy 参考值不一致"
test("跨后端 to_numpy 一致性", test_cross_backend_consistency)

print()

# ---- 7. 设备信息 ----
print("=== 7. 设备信息 ===")

for bk in available:
    def make_test(bk=bk):
        def _test():
            b = get_backend(bk)
            dev = b.device
            print(f"    {bk}: {dev}")
        return _test
    test(f"{bk} 设备信息", make_test())

print()

# ---- 8. 性能基准（简单对比） ----
print("=== 8. 性能基准（简单对比） ===")

N = 100000
for bk in available:
    if bk == 'jax':
        continue  # JAX 只读，跳过
    def make_test(bk=bk, N=N):
        def _test():
            from ecs.engine import RECS
            t0 = time.perf_counter()
            r = RECS(N, {'x': float, 'y': float}, backend=bk)
            r.add(N, x=np.random.randn(N).tolist(), y=np.random.randn(N).tolist())
            t1 = time.perf_counter()
            r.where_set(lambda p: p.get_attr('x') > 0, {'y': -1.0})
            t2 = time.perf_counter()
            print(f"    {bk}: 添加 {N} 实体 {t1-t0:.3f}s, where_set {t2-t1:.3f}s")
        return _test
    test(f"{bk} 性能 ({N} 实体)", make_test())

print()

# ---- 总结 ----
print("=" * 60)
print(f"测试完成: {passed} 通过, {failed} 失败, {skipped} 跳过")
if errors:
    print("\n错误详情:")
    for name, msg, tb in errors:
        print(f"\n  --- {name} ---")
        print(f"  {msg}")
        # print first 3 lines of traceback only
        lines = tb.strip().split('\n')
        for line in lines[-4:]:
            print(f"  {line}")
print("=" * 60)
