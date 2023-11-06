import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rows = np.sum(arr, axis=1)
cols = np.sum(arr, axis=0)

print(rows)
print(cols)