import matplotlib.pyplot as plt


if __name__ == "__main__":
    x = [10,20,40,60,80,100,600,1000,3000]
    y = [0.4169, 0.8049, 0.8329,0.9042,0.9219, 0.9381, 0.9493, 0.9507, 0.9671]
    plt.semilogx(x,y)
    plt.xlabel("Number of labelled samples", fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.ylim([0,1])
    plt.savefig("figures/accuracy_manually.svg")