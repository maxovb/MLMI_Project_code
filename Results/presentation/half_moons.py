from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

n = 10000
std = 0.13
num_labelled = 3 # per class
points, labels = make_moons(n_samples=n, noise=std)
blue_points = points[labels==0]
red_points = points[labels==1]

plt.figure()
plt.plot(blue_points[:num_labelled,0],blue_points[:num_labelled,1],"yv",markersize = 15,alpha=1)
plt.plot(red_points[:num_labelled,0],red_points[:num_labelled,1],"rv",markersize = 15,alpha=1)
plt.axis('off')
plt.xlim([-1.5,2.5])
plt.ylim([-1.1,1.5])
plt.savefig("figures/half_moons_labelled_only.svg")
plt.close()
plt.figure()
plt.axis('off')
plt.plot(points[:,0],points[:,1],"k.",alpha=0.3, markersize=5)
plt.plot(blue_points[:num_labelled,0],blue_points[:num_labelled,1],"yv",markersize = 15,alpha=1)
plt.plot(red_points[:num_labelled,0],red_points[:num_labelled,1],"rv",markersize = 15,alpha=1)
plt.xlim([-1.5,2.5])
plt.ylim([-1.1,1.5])
plt.savefig("figures/half_moons.svg")
plt.close()