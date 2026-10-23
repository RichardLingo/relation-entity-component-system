"""RECS 多后端计算架构——综合边缘测试套件（pytest 化）。

由原 tests/comprehensive_test.py 转换：把自建 test(name, fn) harness 改为 pytest
参数化，断言语义逐条保留。覆盖：
1. 后端注册表
2. 各后端基础 CRUD
3. 字符串列隔离
4. 边缘情况（空池 / 单元素 / 大容量 / 布尔掩码 / remove=disable）
5. Relation 操作
6. 跨后端 to_numpy 导出一致性
7. 各后端设备信息
8. 大池功能性基准（原性能对比段的写路径覆盖）

JAX 数组不可变，所有写操作测试统一跳过 jax（沿用原套件 §2/§3/§6/§8 的处理口径）。
"""

import time

import numpy as np
import pytest

from recs.backends import get_backend, list_backends

AVAILABLE = list_backends()
# jax 数组不可变 → 写操作后端集合排除 jax
WRITE_BACKENDS = [b for b in AVAILABLE if b != "jax"]


def _to_numpy(arr):
    """把任意后端数组转为 NumPy。"""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "cpu") and hasattr(arr, "detach"):  # torch
        return arr.cpu().detach().numpy()
    return np.asarray(arr)


def assert_arr_eq(a, b, msg=""):
    """跨后端数组比较（字符串用 array_equal，数值用 allclose）。"""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if a_np.dtype.kind in ("U", "S") or b_np.dtype.kind in ("U", "S"):
        assert np.array_equal(a_np, b_np), f"{msg}: {a_np} != {b_np}"
    else:
        assert np.allclose(a_np, b_np), f"{msg}: {a_np} != {b_np}"


# ---- 1. 后端注册表 ----
def test_backend_registry():
    assert "numpy" in AVAILABLE, "NumPy 必须是可用后端"
    assert get_backend("numpy").name == "numpy"


# ---- 2. 各后端基础 CRUD ----
@pytest.mark.parametrize("bk", WRITE_BACKENDS)
def test_backend_crud(bk):
    from ecs.engine import EntityPool, RECS

    r = RECS(16, {"x": float, "y": float}, backend=bk)
    r.add(5, x=[1.0, 2.0, 3.0, 4.0, 5.0], y=[10.0, 20.0, 30.0, 40.0, 50.0])
    assert r.size == 5
    assert_arr_eq(r.get_attr("x")[:5], [1.0, 2.0, 3.0, 4.0, 5.0], f"{bk} RECS.add")

    r.where_set(lambda p: p.get_attr("x") > 3, {"x": 99.0})
    assert_arr_eq(r.get_attr("x")[:5], [1.0, 2.0, 3.0, 99.0, 99.0], f"{bk} where_set")

    idxs = r.query(lambda p: p.get_attr("x") == 99.0, return_="indices")
    assert len(idxs) == 2, f"{bk} query indices: expected 2, got {len(idxs)}"

    view = r.query(lambda p: p.get_attr("x") == 99.0, return_="view")
    assert len(view) == 2
    view["y"] = 888.0
    assert_arr_eq(r.get_attr("y")[:5], [10.0, 20.0, 30.0, 888.0, 888.0], f"{bk} view set")

    r.remove(idxs)
    for i in idxs:
        assert not r.is_active(i), f"{bk} remove: entity {i} should be inactive"

    out = r.to_numpy()
    assert isinstance(out["x"], np.ndarray)

    r.disable([0])
    assert not r.is_active(0)
    r.enable([0])
    assert r.is_active(0)

    p = EntityPool(16, {"a": int}, backend=bk)
    p.add(3, a=[10, 20, 30])
    assert p.size == 3
    assert_arr_eq(p.get_attr("a")[:3], [10, 20, 30], f"{bk} EntityPool")


# ---- 3. 字符串列隔离 ----
@pytest.mark.parametrize("bk", WRITE_BACKENDS)
def test_string_column_isolation(bk):
    from ecs.engine import RECS

    r = RECS(16, {"name": "U10", "tag": "U5", "val": float}, backend=bk)
    r.add(4, name=["A", "B", "C", "D"], tag=["x", "y", "z", "w"], val=[1.0, 2.0, 3.0, 4.0])
    assert r.size == 4
    assert isinstance(r.d["name"], np.ndarray)
    assert isinstance(r.d["tag"], np.ndarray)
    assert r.d["name"].dtype.kind == "U"
    if bk == "torch":
        import torch

        assert isinstance(r.d["val"], torch.Tensor)
    r.where_set(lambda p: p.get_attr("val") > 2, {"val": 99.0})
    assert_arr_eq(r.get_attr("name")[:4], ["A", "B", "C", "D"], f"{bk} string unchanged")


# ---- 4. 边缘情况 ----
def test_empty_pool():
    from ecs.engine import RECS

    r = RECS(16, {"x": float})
    assert r.size == 0
    idxs = r.query(return_="indices")
    assert idxs is not None
    assert len(idxs) == 0


def test_single_element():
    from ecs.engine import RECS

    r = RECS(16, {"x": float})
    r.add(1, x=[42.0])
    assert r.size == 1
    assert r.get_attr("x")[0] == 42.0
    r.where_set(lambda p: p.get_attr("x") == 42.0, {"x": 0.0})
    assert r.get_attr("x")[0] == 0.0


def test_large_capacity():
    from ecs.engine import RECS

    r = RECS(1024, {"x": float})
    r.add(1000, x=np.random.randn(1000).tolist())
    assert r.size == 1000
    assert len(r.d["x"]) >= 1000


def test_bool_mask_query():
    from ecs.engine import RECS

    r = RECS(16, {"x": float})
    r.add(5, x=[1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, False, True, False, True])
    idxs = r.query(mask, return_="indices")
    assert len(idxs) == 3


def test_remove_disable():
    """remove = disable，size 不变但标记为 inactive。"""
    from ecs.engine import RECS

    r = RECS(16, {"x": float})
    r.add(3, x=[1.0, 2.0, 3.0])
    r.remove([0, 1, 2])
    assert r.size == 3  # remove 不改变 size
    assert not r.is_active(0)
    assert not r.is_active(1)
    assert not r.is_active(2)


# ---- 5. Relation 操作 ----
@pytest.mark.parametrize("bk", WRITE_BACKENDS)
def test_relation(bk):
    from ecs.engine import Relation

    rel = Relation("test", capacity=32, attr_dtypes={"weight": float, "label": "U5"}, backend=bk)
    rel.add([0, 1, 2], [1, 2, 0], weight=[0.5, 0.8, 0.3], label=["a", "b", "c"])
    assert rel.size == 3
    assert_arr_eq(rel.get_attr("weight")[:3], [0.5, 0.8, 0.3], f"{bk} Relation.add")
    assert isinstance(rel.get_attr("label")[:3], np.ndarray)

    dsts, _ = rel.neighbors_from_uid(0)
    assert len(dsts) == 1

    rel.remove_by_indices([1])
    assert rel.size == 2

    order = rel.argsort_by("weight")
    assert len(order) == rel.size


# ---- 6. to_numpy 导出一致性 ----
@pytest.mark.parametrize("bk", WRITE_BACKENDS)
def test_to_numpy_consistency(bk):
    from ecs.engine import RECS

    np.random.seed(42)
    data = np.random.randn(10)
    r = RECS(32, {"x": float}, backend=bk)
    r.add(10, x=data.tolist())
    out = r.to_numpy()
    assert isinstance(out["x"], np.ndarray)
    assert np.allclose(out["x"], data), f"{bk} to_numpy 数值不一致"


def test_cross_backend_consistency():
    from ecs.engine import RECS

    np.random.seed(42)
    data = np.random.randn(10)
    ref = None
    for bk in WRITE_BACKENDS:
        r = RECS(32, {"x": float}, backend=bk)
        r.add(10, x=data.tolist())
        out = r.to_numpy()
        if ref is None:
            ref = out["x"].copy()
        else:
            assert np.allclose(ref, out["x"]), f"{bk} 与 NumPy 参考值不一致"


# ---- 7. 设备信息 ----
@pytest.mark.parametrize("bk", AVAILABLE)
def test_device_info(bk):
    dev = get_backend(bk).device
    assert dev is not None


# ---- 8. 大池功能性基准（原性能对比段的写路径覆盖）----
@pytest.mark.parametrize("bk", WRITE_BACKENDS)
def test_large_pool_perf_path(bk):
    from ecs.engine import RECS

    N = 100000
    t0 = time.perf_counter()
    r = RECS(N, {"x": float, "y": float}, backend=bk)
    r.add(N, x=np.random.randn(N).tolist(), y=np.random.randn(N).tolist())
    t1 = time.perf_counter()
    r.where_set(lambda p: p.get_attr("x") > 0, {"y": -1.0})
    t2 = time.perf_counter()
    assert r.size == N
    # 仅记录耗时，不做时间断言（避免机器差异导致 flaky）
    assert t2 >= t1 >= t0
