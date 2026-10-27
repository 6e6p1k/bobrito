import sys
sys.path.insert(0, 'python')

import numpy as np 
import bobrito as bobr
a = bobr.arange(0, 30).reshape([3, 10])
a.eval()

bobr.mem_snapshot()

np_a = np.full([3, 10], 1, dtype=np.float32)
print(np_a)
print(a)

bobr.mem_snapshot()
bobr.sync()

a.steal_data(np_a)
print(a)
bobr.mem_snapshot()

# [VRAM SUMMARY]
#   CUDA: 1027 MB used / 11769 MB total (10741 MB free)
#   Pool: used: 1, bytes used: 1024

# [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
# Node(label='arange(start=0, end=30, step=1).reshape()', shape=[3, 10], data:
# [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
# [VRAM SUMMARY]
#   CUDA: 1027 MB used / 11769 MB total (10741 MB free)
#   Pool: used: 1, bytes used: 1024

# Node(label='arange(start=0, end=30, step=1).reshape()', shape=[3, 10], data:
# [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# [VRAM SUMMARY]
#   CUDA: 1027 MB used / 11769 MB total (10741 MB free)
#   Pool: used: 1, bytes used: 1024