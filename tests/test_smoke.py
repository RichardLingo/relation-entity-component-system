"""RECS 冒烟测试（pytest 化）。

由原 tests/smoke_test.py 转换：把 print 观测点固化为断言，语义不变——
覆盖 add / query(view) / 视图写回 / where_set / dense_matrix_to_relation / 边视图写回。
"""

import numpy as np

from recs import RECS


def test_smoke_add_and_size():
    engine = RECS(16, {"hp": "float64", "type": "int32"})
    assert len(engine) == 0

    idxs = engine.add(3, hp=[10.0, 20.0, 30.0], type=[1, 2, 1])
    assert len(idxs) == 3
    assert len(engine) == 3
    # uid_array 应覆盖全部活跃实体
    assert len(engine.uid_array()) >= 3


def test_smoke_view_writeback():
    engine = RECS(16, {"hp": "float64", "type": "int32"})
    engine.add(3, hp=[10.0, 20.0, 30.0], type=[1, 2, 1])

    # type == 1 的实体为索引 0、2
    view = engine.query(lambda p: p.get_attr("type") == 1, return_="view")
    assert sorted(view.indices().tolist()) == [0, 2]

    # 经视图写回应穿透到引擎主存
    view["hp"] = 99.0
    hp = np.asarray(engine.get_attr("hp"))[:3]
    assert np.allclose(hp, [99.0, 20.0, 99.0])


def test_smoke_where_set():
    engine = RECS(16, {"hp": "float64", "type": "int32"})
    engine.add(3, hp=[10.0, 20.0, 30.0], type=[1, 2, 1])
    view = engine.query(lambda p: p.get_attr("type") == 1, return_="view")
    view["hp"] = 99.0  # hp -> [99, 20, 99]

    # hp < 50 的仅索引 1（20.0）→ type=9, hp=50
    engine.where_set(lambda p: p.get_attr("hp") < 50, {"type": 9, "hp": 50.0})
    hp = np.asarray(engine.get_attr("hp"))[:3]
    tp = np.asarray(engine.get_attr("type"))[:3]
    assert np.allclose(hp, [99.0, 50.0, 99.0])
    assert tp.tolist() == [1, 9, 1]


def test_smoke_dense_matrix_to_relation():
    engine = RECS(16, {"hp": "float64", "type": "int32"})
    engine.add(3, hp=[10.0, 20.0, 30.0], type=[1, 2, 1])

    mat = np.array([[0.0, 1.5, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.3]])
    rel = engine.dense_matrix_to_relation(
        mat, relation_name="rtest", attr_name="w", threshold=1.0
    )
    # 阈值 1.0 之上的非零项：1.5(0→1)、2.0(1→0)、3.3(2→2) 共 3 条边
    assert rel.size == 3

    # 边视图写回穿透到关系主存
    edge_view = rel.query(return_="view")
    assert len(edge_view) == 3
    edge_view["w"] = 7.7
    w = np.asarray(rel.get_attr("w"))[: rel.size]
    assert np.allclose(w, 7.7)
