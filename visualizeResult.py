
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#######################################
##
##   Visualize clustering results
##
#######################################

loglik = np.loadtxt("dat\loglik")
plt.plot(loglik)
plt.title("Log likelihood")
plt.xlabel("iteration")
plt.savefig("img\Log_likelihood.png")

clstr = stats.mode(np.loadtxt("dat\z").T).mode

D = np.loadtxt("dat\data")
theta = np.angle(D[:,0] + D[:,1] * 1j).reshape(-1,1)
x = D[:,-1].reshape(-1,1)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(np.cos(theta), np.sin(theta), x, c = clstr)
plt.title("Clustering Result")
ax.view_init(30, 60)
plt.savefig("img\Clustering_Result.png")


