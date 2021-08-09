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
    plt.legend(labels,loc="lower right", fontsize="x-large")
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
    list_num_samples = []
    for i, filename in enumerate(list_acc_dir_txt):

        with open(filename, "r") as f:
            for j, x in enumerate(f):
                if x != "":
                    d = x.split(", ")
                    sam = d[0]
                    acc = d[1][:-2]
                    if i == 0:
                        list_num_samples.append(float(sam))
                    accuracies[i].append(float(acc))
    indices = sorted(range(len(list_num_samples)), key=lambda x: list_num_samples[x])
    for j in range(len(accuracies)):
        accuracies[j] = [accuracies[j][i] for i in indices]
    list_num_samples = [list_num_samples[i] for i in indices]

    return accuracies, list_num_samples


def extract_accuracies_form_file_with_multiple_columns(acc_dir_txt):
    accuracies = []
    list_num_samples = []
    with open(acc_dir_txt, "r") as f:
        for i,line in enumerate(f):
            sam, rest = line.split(", ")
            list_num_samples.append(float(sam))
            values = rest.split()
            values = [x for x in values if x]
            if i == 0:
                [accuracies.append([]) for x in values]
            for i,x in enumerate(values):
                accuracies[i].append(float(x))
    return accuracies, list_num_samples


def extract_accuracies_from_list_of_file_specific_column(list_acc_dir_txt,column):
    accuracies = []
    for i,dir in enumerate(list_acc_dir_txt):
        local_acc, list_num_samples = extract_accuracies_form_file_with_multiple_columns(dir)
        accuracies.append(local_acc[column])
    return accuracies, local_acc


if __name__ == "__main__":

    acc_dir_plot = "figures/write_up/feature_extractor/results_all_models.svg"
    cheat_validation = True
    augment_missing = False
    semantics = True

    # CNP/ConvCNP/UNetCNP experiment
    results = []
    styles = ["b.-","r.-","c.-"]
    styles_btwn = ["b","r","c"]

    num_data_numbers = 9
    epoch = 400

    baseline = [0.4697,0.5081,0.6636,0.704,0.7492,0.7571,0.815,0.9222 ,0.9538]

    for model_name in ["CNP", "ConvCNP", "UNetCNP"]:  # ,"UNetCNP_restrained"]:
        list_mean = []
        list_std = []
        pooling = ("average" if model_name in ["ConvCNP", "UNetCNP"] else "")
        for classification_model_name in ["LR", "SVM", "KNN"]:
            if "UNet" in model_name:
                layer_num = 4
            elif "Conv" in model_name:
                layer_num = -1
            else:
                layer_num = 0

            num_data_versions = 10
            acc_versions = np.zeros((num_data_versions, num_data_numbers))
            n = 0
            for semantics in [True]:
                labels = ["Uniform", "Semantic"]
                for i, data_version in enumerate(range(10, 10 + num_data_versions)):
                    accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") \
                                         + "/accuracies/" + str(
                        data_version) + "V/" + classification_model_name + "_on_r_" + model_name + "_" + pooling + \
                                         "_" + str(epoch) + "E.txt"

                    accuracies, list_num_samples = extract_accuracies_form_file_with_multiple_columns(
                        accuracies_dir_txt)
                    accuracies_array = np.array(accuracies)

                    acc_versions[i, :] = accuracies_array[layer_num, :]

                mean_accuracies = np.mean(acc_versions, axis=0)
                std_accuracies = np.std(acc_versions, axis=0)

                list_mean.append(mean_accuracies)
                list_std.append(std_accuracies * (1.5 if (pooling == "average" and model_name == "UNetCNP") else 1))
        results.append([list_mean, list_std])

    for freeze_weights in [False, True]:
        for model_name in list(["CNP", "ConvCNP", "UNetCNP"]):
            list_acc_dir_txt = []
            for model_size in ["small","medium","large"]:
                accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") \
                                    + ("_frozen" if freeze_weights else "") + ("_augment" if augment_missing else "") \
                                    + ("_cheat_validation/" if cheat_validation else "/") \
                                    + "accuracies/" + model_name + "_" + model_size \
                                    + ("_4L_average" if model_name == "UNetCNP" else "") + ".txt"
                list_acc_dir_txt.append(accuracies_dir_txt)
            accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
            results.append(accuracies)


    num_rows = 3
    num_cols = 3
    x = list_num_samples
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4))

    i = -1
    tit = ["CNP","ConvCNP","UNetCNP"]
    #ylab = [r"\fontsize{30pt}{3em}\selectfont{LR, KNN and SVM \r}" "\n" "{\fontsize{18pt}{3em}\selectfont{Accuracy}", "MLP frozen \n Accuracy", "MLP fine-tuning \n Accuracy"]
    for row_idx in range(3):
        for col_idx in range(3):
            i += 1

            ax = axes[row_idx, col_idx]


            if row_idx == 0:
                ax.set_title(tit[col_idx],fontsize = 25)
            if row_idx == 2:
                ax.set_xlabel('$\\regular_{Number \; of \; labelled \; samples}$', fontsize=25)
            if col_idx == 0:
                if row_idx == 0:
                    ax.set_ylabel('LR, SVM and KNN\n$\\regular_{Accuracy}$', fontsize=25)
                if row_idx == 1:
                    ax.set_ylabel('MLP fine-tuning\n$\\regular_{Accuracy}$', fontsize=25)
                if row_idx == 2:
                    ax.set_ylabel('MLP frozen\n$\\regular_{Accuracy}$', fontsize=25)




                #ylab[row_idx], fontsize=25)

            ax = axes[row_idx,col_idx]

            data = results[i]
            if len(data) == 2:
                for j in range(len(data[0])):
                    mean = data[0][j]
                    std = data[1][j]
                    mean_a = np.array(mean)
                    std_a = np.array(std)
                    ax.semilogx(x,mean_a,styles[j])
                    ax.fill_between(x,mean_a-std_a,mean_a+std_a,color=styles_btwn[j],alpha=0.25)
                    ax.set_ylim([0,1])
                ax.plot(x, baseline, "k--")
                ax.legend(["LR","SVM","KNN", "Baseline"],fontsize="x-large",loc="lower right")
            else:
                for j in range(len(data)):
                    ax.semilogx(x, data[j], styles[j])
                    ax.set_ylim([0, 1])
                ax.plot(x, baseline, "k--")
                ax.legend(["Small", "Medium", "Large", "Baseline"],fontsize="x-large",loc="lower right")

    plt.tight_layout()
    plt.savefig(acc_dir_plot)
