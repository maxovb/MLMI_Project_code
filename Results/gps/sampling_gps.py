import stheno
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    kernels = [stheno.EQ().stretch(1) * 0.1, stheno.EQ().stretch(0.1) * 1]
    gps = [stheno.GP(kernel) for kernel in kernels]
    labels = [""]
    num_samples = 5
    step = 0.01
    x = np.arange(-2,2 + step, step)

    dir_plot = "../figures/write_up/gps/gp.svg"
    for i in range(len(gps)):
        for j in range(num_samples):
            y = gps[i](x).sample()
            plt.plot(x,y,alpha=0.5,color="b")
            plt.ylim([-4,4])
            plt.xlabel("x",fontsize=15)
            plt.ylabel("y",fontsize=15)
        dir_plot_local = dir_plot[:-4] + str(i) + dir_plot[-4:]
        plt.savefig(dir_plot_local)
        plt.close()