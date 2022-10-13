
import numpy as np
import matplotlib.pyplot as plt

#######################################
##
##   Create circular-linear data
##
#######################################

N = 500
D1 = np.random.multivariate_normal([0, 0, 4], [[1, 0, -0.8], 
                                               [0, 1, 0], 
                                               [-0.8, 0, 1]], N)
D2 = np.random.multivariate_normal([2, 0, -2], [[1, 0, 0], 
                                               [0, 1, -0.8], 
                                               [0, -0.8, 1]], N)
D3 = np.random.multivariate_normal([-2, 0.2, 0], [[1, 0, 0], 
                                               [0, 1, 0], 
                                               [0, 0, 1]], N)
D = np.concatenate([D1,D2,D3], axis = 0)
np.savetxt("dat\data", D)


#######################################
##
##   Visualize data
##
#######################################

theta = np.angle(D[:,0] + D[:,1] * 1j).reshape(-1,1)
x = D[:,-1].reshape(-1,1)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(np.cos(theta), np.sin(theta), x, c = [1]*N + [2]*N + [3]*N)
plt.title("3D Scatter Plot")
ax.view_init(30, 60)
plt.savefig("img\Data_scatter.png")



