import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":


    x = [60000 / 16 * 0.9 / 100 / 2**i for i in range(6)]
    y_GN = [0.9441, 0.9353, 0.9406, 0.8843, 0.9158, 0.9359]

    y = [0.8676, 0.8839, 0.8619, 0.8934, 0.8846, 0.8909]

    y_ET = [0.9005,0.8308]
    x_ET = [60000 / 16 * 0.9 / 100 / 2**i for i in [0,5]]


    dir_plot = "tunin_sup_weight_test.svg"
    plt.figure()
    plt.plot(x, y,"-o")
    plt.plot(x,y_GN,"-o")
    plt.plot(x_ET,y_ET,"o")
    plt.legend(["Consistency loss","Consistency loss + GradNorm","Extra task + GradNorm"],fontsize="x-large")
    plt.xlabel("Relative weigth of the supervised task",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.plot()
    plt.ylim([0,1])
    plt.savefig(dir_plot)