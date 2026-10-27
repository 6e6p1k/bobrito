import sys
sys.path.insert(0, 'python')
import numpy as np
import bobrito as bobr
shape = [4, 3]

a = np.random.rand(*shape)
a_b = bobr.empty(shape)
a_b.data_from_numpy(a)
# a_b = bobr.Node(a)

print(a)
print(a_b.data)

assert np.allclose(a, np.array(a_b.data))