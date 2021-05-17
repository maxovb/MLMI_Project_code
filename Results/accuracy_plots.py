import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(list_acc_dir_txt, acc_dir_plot, labels, styles=None):
    # input processing
    l = len(list_acc_dir_txt)
    assert l == len(labels), "Ensure that the number of labels corresponds to the number of accuracy text files"

    # loading the accuracies
    accuracies = [[] for _ in range(l)]
    for i,filename in enumerate(list_acc_dir_txt):
        if i ==0: # get the number of samples used for each accuracy entry
            with open(filename, "r") as f:
                list_num_samples = (f.readlines()[0].split(":")[1]).split()
                list_num_samples = [x for x in list_num_samples if x]
                list_num_samples = np.array(list_num_samples).astype(int)

        with open(filename,"r") as f:
            for j,x in enumerate(f):
                if j != 0: # first line does not contain the accuracies, it stores the number of samples
                    if x != "":
                        accuracies[i].append(float(x))

    # plot
    plt.figure()
    for i in range(l):
        if not(styles):
            plt.plot(list_num_samples,accuracies[i])
        else:
            plt.plot(list_num_samples, accuracies[i],styles[i])
    plt.legend(labels)
    plt.xlabel("Number of labelled samples",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.xscale("log")
    plt.savefig(acc_dir_plot)


if __name__ == "__main__":

    cheat_validation = True

    # CNP/ConvCNP experiments
    freeze_weights = False
    augment_missing = False
    semantics = True
    acc_dir_plot = "figures/accuracies_supervised" + ("_semantics" if semantics else "")\
                   + ("_frozen" if freeze_weights else "") + ("_augment" if augment_missing else "") \
                   + ("_cheat_validation.svg" if cheat_validation else ".svg")
    styles = ["r-","r--","r-.","b-","b--","b-."]

    list_acc_dir_txt = []
    labels = []
    for model_name in ["CNP","ConvCNP"]:
        for model_size in ["small","medium","large"]:
            labels.append(model_name + " " + model_size)
            accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") \
                                + ("_frozen" if freeze_weights else "") + ("_augment" if augment_missing else "") \
                                + ("_cheat_validation/" if cheat_validation else "/") \
                                + "accuracies/" + model_name + "_" + model_size + ".txt"
            list_acc_dir_txt.append(accuracies_dir_txt)

    plot_accuracy(list_acc_dir_txt, acc_dir_plot, labels, styles=styles)

    # KNN baseline
    acc_dir_plot = "figures/accuracies_KNN.svg"
    accuracies_dir_txt = "../saved_models/MNIST/supervised/accuracies/KNN.txt"
    styles_knn = ["r-"]
    labels = ["KNN"]
    list_acc_dir_txt = [accuracies_dir_txt]
    plot_accuracy(list_acc_dir_txt, acc_dir_plot, labels, styles=styles_knn)

    # LeNet baseline
    acc_dir_plot = "figures/accuracies_LeNet" + ("_cheat_validation.svg" if freeze_weights else ".svg")

    list_acc_dir_txt = []
    labels = []
    for model_size in ["small", "medium", "large"]:
        labels.append("LeNet " + model_size)
        accuracies_dir_txt = "../saved_models/MNIST/supervised" + (
                "_cheat_validation/" if cheat_validation else "/") + "accuracies/" + "LeNet_" + model_size + ".txt"
        list_acc_dir_txt.append(accuracies_dir_txt)
    plot_accuracy(list_acc_dir_txt, acc_dir_plot, labels, styles=styles)


    # KNN and LR on the representation out of the encoder
    acc_dir_plot = "figures/accuracies_supervised" + ("_semantics" if semantics else "") + "_KNN_LR_on_r.svg"
    
    styles_r = ["r","b"]
    list_acc_dir_txt = []
    labels = []
    for classification_model_name in ["KNN","LR"]:
        labels.append(classification_model_name)
        accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + "/accuracies/" + classification_model_name + "_on_r.txt"
        list_acc_dir_txt.append(accuracies_dir_txt)
    plot_accuracy(list_acc_dir_txt, acc_dir_plot, labels, styles=styles_r)








