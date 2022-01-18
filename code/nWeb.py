import numpy as np
import torch as th
import GPUtil

# l = [[1, 4, 5, 6], [2, 5, 7, 88]]
# a = th.FloatTensor(l)
# print(f"{a.shape}")
#
# a = th.IntTensor(2, 3, 3)
# print(a)
# print(a.is_cuda)
#
# a = [[-5, 5], [2, 3], [1, -1]]
# b = [-0.5, 2.5]
# a=np.array(a)
# b=np.array(b)
# print(a*b)

a = th.IntTensor(1000, 100, 5)
a = a.cuda()
print(a)
print(a.is_cuda)
print(GPUtil.showUtilization())
a = a.cpu()
th.cuda.empty_cache()
print(GPUtil.showUtilization())

