"""RECS 核心实现包。

说明
----
核心实现收敛到 `engine.py`。
推荐用法：

- `from recs import RECS, Relation`
"""

from .engine import (
    RECS,
    EntityPool,
    EntityPoolView,
    Relation,
    RelationView,
    DenseRelation,
    SparseRelationEdges,
    SparseBoolBitset,
    SparseNullable,
    dense_to_relation,
    to_indices,
    to_mask,
)

__all__ = [
    'RECS',
    'EntityPool',
    'EntityPoolView',
    'Relation',
    'RelationView',
    'DenseRelation',
    'SparseRelationEdges',
    'SparseBoolBitset',
    'SparseNullable',
    'dense_to_relation',
    'to_indices',
    'to_mask',
]
