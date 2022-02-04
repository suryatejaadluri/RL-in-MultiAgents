import numpy as np

Q = np.ones((8,8,2,5,5))
Q[0][1][1][0][1] = 52
Q[0][1][1][0][2] = 52
print(np.argmax(Q[0][1][1])//5)

maxid = np.where(Q[0][1][1] == np.max(Q[0][1][1]))
pass