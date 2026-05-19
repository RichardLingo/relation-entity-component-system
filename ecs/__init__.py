"""ECS 子模块（单文件实现）。

说明
----
本包的核心实现已收敛到 `ecsengine.py`。
推荐用法：

- `from ecs import ECSEngine, Relation`
"""

from .ecsengine import (
    ECSEngine,
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
    'ECSEngine',
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
