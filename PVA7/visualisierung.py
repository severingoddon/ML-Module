import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = pd.read_csv('spiraldatensatz.csv')

x = data['x1']
y = data['x2']
z = data['x3']

ax.scatter(x, y, z, c='r', marker='o', s=0.1)
ax.view_init(elev=32, azim=-112) # rotate the 3d model to see the spiral

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()