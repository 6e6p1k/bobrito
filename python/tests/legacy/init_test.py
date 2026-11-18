import sys
sys.path.insert(0, 'python')
import torch
import numpy 


shape = [2, 3, 4]
a = bobr.kaiming_normal(shape)
a.eval()
print(a.data)