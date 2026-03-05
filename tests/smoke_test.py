import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from ecs.ecsengine import ECSEngine

print('SMOKE TEST START')
engine = ECSEngine(16, {'hp': 'float64', 'type': 'int32'})
print('engine.size before', len(engine))

# add entities
idxs = engine.add(3, hp=[10.0, 20.0, 30.0], type=[1, 2, 1])
print('added idxs', idxs)
print('size after add', len(engine))
print('uid_array', engine.uid_array())

# query view and set via view
view = engine.query(lambda p: p.get_attr('type') == 1, return_='view')
print('view indices', view.indices())
print('view hp before', view['hp'])
view['hp'] = 99.0
print('hp after setting view to 99', engine.get_attr('hp'))

# where_set
engine.where_set(lambda p: p.get_attr('hp') < 50, {'type': 9, 'hp': 50.0})
print('hp after where_set (<50 -> 50)', engine.get_attr('hp'))
print('type after where_set', engine.get_attr('type'))

# dense matrix to relation
mat = np.array([[0.0, 1.5, 0.0],[2.0, 0.0, 0.0],[0.0, 0.0, 3.3]])
rel = engine.dense_matrix_to_relation(mat, relation_name='rtest', attr_name='w', threshold=1.0)
print('rel size', rel.size)
if rel.size > 0:
    print('rel src', rel.d['src_uid'][:rel.size])
    print('rel dst', rel.d['dst_uid'][:rel.size])
    print('rel w before', rel.get_attr('w'))
    # view assign on edges
    edge_view = rel.query(return_='view')
    print('edge_view len', len(edge_view))
    edge_view['w'] = 7.7
    print('rel w after set', rel.get_attr('w')[:rel.size])

print('SMOKE_OK')
