import sys

sys.path.append("./build")
import assign_cost_cuda
import numpy as np
import time

data_left = np.array(
    [
        [-1, 1, 3],
        [-1, 3, 4],
        [5, 3, -2],
        [2, 0, -1],
    ]
)
batch_built = np.array(
    [
        [-1, 1, 2, 3, -2, -2],
        [-1, 3, 5, 4, -1, -2],
        [5, 3, 1, -2, -2, -2],
        [2, 0, 1, -1, -2, -2],
    ]
)
# for i in range(50000):
ac = assign_cost_cuda.assign_cost(data_left, batch_built, 4, 6)
print(ac)
