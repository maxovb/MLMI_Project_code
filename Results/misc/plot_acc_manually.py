import matplotlib.pyplot as plt


if __name__ == "__main__":
    x = [10,20,40,60,80,100,600,1000,3000]
    baseline = [0.4697,0.5081,0.6636,0.704,0.7492,0.7571,0.815,0.9222 ,0.9538]
    y_feature_extractor =[0.5335,0.6417,0.669,0.7563,0.7751,0.8079,0.9252,0.9425,0.9634]

    y = [0.4169, 0.8049, 0.8329,0.9042,0.9219, 0.9381, 0.9493, 0.9507, 0.9671]

    plt.semilogx(x,baseline,"k--",label="Baseline")
    plt.semilogx(x,y_feature_extractor,"C0--.",label="UNetCNP + medium MLP")
    plt.semilogx(x, y, "C1.-", label="UNetCNP + CL + GN")
    plt.legend(fontsize=15)
    plt.xlabel("Number of labelled samples", fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.ylim([0,1])
    plt.savefig("../figures/write_up/joint_NP/joint_NP_accuracy_vs_num_labelled.svg")