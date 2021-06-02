import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=None):
    # input processing
    l = len(accuracies)
    assert l == len(labels), "Ensure that the number of labels corresponds to the number of accuracy text files"

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
    plt.ylim([0,1])
    plt.savefig(acc_dir_plot)


def plot_accuracy_vs_layer_num(accuracies, layer_numbers, acc_dir_plot, labels, styles=None):
    # input processing
    l = len(accuracies)
    assert l == len(labels), "Ensure that the number of labels corresponds to the number of accuracy text files"

    # plot
    plt.figure()
    for i in range(l):
        if not (styles):
            plt.plot(layer_numbers, accuracies[i])
        else:
            plt.plot(layer_numbers, accuracies[i], styles[i])
    plt.legend(labels)
    plt.xlabel("Layer Number", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim([0, 1])
    plt.savefig(acc_dir_plot)


def extract_accuracies_from_list_of_files(list_acc_dir_txt):
    # loading the accuracies
    accuracies = [[] for _ in range(len(list_acc_dir_txt))]
    for i, filename in enumerate(list_acc_dir_txt):
        if i == 0:  # get the number of samples used for each accuracy entry
            with open(filename, "r") as f:
                list_num_samples = (f.readlines()[0].split(":")[1]).split()
                list_num_samples = [x for x in list_num_samples if x]
                list_num_samples = np.array(list_num_samples).astype(int)

        with open(filename, "r") as f:
            for j, x in enumerate(f):
                if j != 0:  # first line does not contain the accuracies, it stores the number of samples
                    if x != "":
                        accuracies[i].append(float(x))
    return accuracies, list_num_samples


def extract_accuracies_form_file_with_multiple_columns(acc_dir_txt):
    accuracies = []
    with open(acc_dir_txt, "r") as f:
        for i,line in enumerate(f):
            values = line.split()
            values = [x for x in values if x]
            if i == 0: # initialize the lsit of accuracies
                list_num_samples = line.split(":")[1].split()
                list_num_samples = [x for x in list_num_samples if x]
                list_num_samples = np.array(list_num_samples).astype(int)
            else:
                if i == 1:
                    [accuracies.append([]) for x in values]
                for i,x in enumerate(values):
                    accuracies[i].append(float(x))
    return accuracies, list_num_samples



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

    accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles)

    # LR baseline
    acc_dir_plot = "figures/accuracies_LR.svg"
    accuracies_dir_txt = "../saved_models/MNIST/supervised/accuracies/LR.txt"
    styles_knn = ["r-"]
    labels = ["LR"]
    list_acc_dir_txt = [accuracies_dir_txt]

    accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_knn)

    """
    # KNN baseline
    acc_dir_plot = "figures/accuracies_KNN.svg"
    accuracies_dir_txt = "../saved_models/MNIST/supervised/accuracies/KNN.txt"
    styles_knn = ["r-"]
    labels = ["KNN"]
    list_acc_dir_txt = [accuracies_dir_txt]

    accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_knn)

    # LeNet baseline
    acc_dir_plot = "figures/accuracies_LeNet" + ("_cheat_validation.svg" if freeze_weights else ".svg")

    list_acc_dir_txt = []
    labels = []
    for model_size in ["small", "medium", "large"]:
        labels.append("LeNet " + model_size)
        accuracies_dir_txt = "../saved_models/MNIST/supervised" + (
                "_cheat_validation/" if cheat_validation else "/") + "accuracies/" + "LeNet_" + model_size + ".txt"
        list_acc_dir_txt.append(accuracies_dir_txt)

    accuracies, list_num_samples= extract_accuracies_from_list_of_files(list_acc_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles)


    # KNN and LR on the representation out of the encoder
    acc_dir_plot = "figures/accuracies_supervised" + ("_semantics" if semantics else "") + "_KNN_LR_on_r.svg"
    
    styles_r = ["r","b"]
    list_acc_dir_txt = []
    labels = []
    for classification_model_name in ["KNN","LR"]:
        labels.append(classification_model_name)
        accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + "/accuracies/" + classification_model_name + "_on_r.txt"
        list_acc_dir_txt.append(accuracies_dir_txt)

    accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles)
    """

    for model_name in ["ConvCNP","UNetCNP","UNetCNP_restrained"]:
        for pooling in ["average","flatten"]:
            epoch = 400
            # KNN and LR on the representation out of the different layers
            for classification_model_name in ["KNN", "LR"]:
                accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") \
                                     + "/accuracies/" + classification_model_name + "_on_r_" + model_name + "_" + pooling + \
                                     "_"+ str(epoch) + "E.txt"
                acc_dir_plot = "figures/" + classification_model_name + "/accuracies_supervised" + \
                               ("_semantics" if semantics else "") + "_" + classification_model_name + "_on_r_" + model_name \
                               + "_" + pooling + "_"+ str(epoch) + "E.svg"

                dir_to_create = os.path.dirname(acc_dir_plot)
                os.makedirs(dir_to_create,exist_ok=True)

                accuracies, list_num_samples = extract_accuracies_form_file_with_multiple_columns(accuracies_dir_txt)
                labels = []
                for i in range(len(accuracies)):
                    labels.append("Layer " + str(i))
                plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels)

                # accuracy vs layer num
                acc_dir_plot = "figures/" + classification_model_name + "/accuracies_supervised" + \
                               ("_semantics" if semantics else "") + "_" + classification_model_name + "_on_r_" \
                               + model_name + "_" + pooling + "_" + str(epoch) + "E_vs_layer_num.svg"
                accuracies_array = np.array(accuracies)
                accuracies_array = accuracies_array.T
                accuracies_transposed = accuracies_array.tolist()
                labels = []
                for x in list_num_samples:
                    labels.append(str(x) + " samples")
                plot_accuracy_vs_layer_num(accuracies_transposed, list(range(len(accuracies))), acc_dir_plot, labels)












