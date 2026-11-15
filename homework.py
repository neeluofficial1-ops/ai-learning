import numpy as np

marks = np.array([72,85,90,64,88])

print("average:",marks.mean())
print("highest:",marks.max())
print("lowest:",marks.min())
print("80+marks:",marks[marks >= 80])