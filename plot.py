import matplotlib.pyplot as plt
import numpy as np

fp = open("results.txt", "r")

results = fp.readlines()
x = np.arange(0, len(results))
y = np.array(results).astype(int)

plt.scatter(x, y)

plt.xlabel("Episodes")
plt.ylabel("Score")


plt.show()