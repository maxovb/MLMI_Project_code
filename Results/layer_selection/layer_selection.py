import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy_vs_layer_num_with_std(accuracies, layer_numbers, std_accuracies, acc_dir_plot, labels=None, styles=None):

    # plot
    plt.figure()
    for i in range(len(accuracies)):
        if not (styles):
            plt.plot(layer_numbers, accuracies[i],label=labels[i])
            plt.fill_between(layer_numbers,accuracies[i] - std_accuracies[i], accuracies[i] + std_accuracies[i],alpha=0.25)
        else:
            plt.plot(layer_numbers, accuracies[i], styles[i], label= labels[i])
            plt.fill_between(layer_numbers, accuracies[i] - std_accuracies[i], accuracies[i] + std_accuracies[i], c = styles[i][0],alpha=0.25)
    plt.plot([layer_numbers[0],layer_numbers[-1]],[0.7571,0.7571],"k--",label="Baseline",alpha=0.5)
    plt.legend(fontsize="x-large")
    plt.xlabel("Layer Number", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim([0, 1])
    plt.xticks(layer_numbers)
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
    epoch = 400
    semantics=True
    for model_name in ["UNetCNP","ConvCNP"]:#,"UNetCNP_restrained"]:
        for pooling in ["average","flatten"]:
            # KNN and LR on the representation out of the different layers
            for classification_model_name in ["SVM"]:
                if "UNet" in model_name:
                    num_layers = 9
                elif "Conv" in model_name:
                    num_layers = 4
                num_data_versions = (10 if pooling == "average" and model_name == "UNetCNP" else 3)
                acc_versions = np.zeros((num_data_versions,num_layers))
                n = 0
                list_mean = []
                list_std = []
                for semantics in [False,True]:
                    labels = ["Uniform","Semantic"]
                    for i,data_version in enumerate(range(10,10 + num_data_versions)):
                        accuracies_dir_txt = "../saved_models/MNIST/supervised" + ("_semantics" if semantics else "") \
                                             + "/accuracies/" + str(data_version) + "V/" + classification_model_name + "_on_r_" + model_name + "_" + pooling + \
                                             "_"+ str(epoch) + "E.txt"
                        # accuracy vs layer num
                        acc_dir_plot = "figures/" + classification_model_name + "/accuracies_supervised" + \
                                       ("_semantics" if semantics else "") + "_" + classification_model_name + "_on_r_" \
                                       + model_name + "_" + pooling + "_" + str(epoch) + "E_vs_layer_num.svg"

                        dir_to_create = os.path.dirname(acc_dir_plot)
                        os.makedirs(dir_to_create,exist_ok=True)

                        accuracies, list_num_samples = extract_accuracies_form_file_with_multiple_columns(accuracies_dir_txt)
                        accuracies_array = np.array(accuracies)

                        accuracies_array = accuracies_array.T
                        if pooling == "flatten" and model_name == "ConvCNP":
                            acc_versions[i, :] = accuracies_array[0]
                        else:
                            acc_versions[i,:] = accuracies_array[5]

                    mean_accuracies = np.mean(acc_versions,axis=0)
                    std_accuracies = np.std(acc_versions,axis=0)
                    layer_numbers = list(range(num_layers))

                    list_mean.append(mean_accuracies)
                    list_std.append(std_accuracies * (1.5 if (pooling == "average" and model_name == "UNetCNP") else 1))


                plot_accuracy_vs_layer_num_with_std(list_mean, list(range(len(accuracies))), list_std, acc_dir_plot, labels=labels)
                print(acc_dir_plot)














