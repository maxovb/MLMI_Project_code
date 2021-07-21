import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1,"../")
from accuracy_plots import extract_accuracies_from_list_of_files, extract_accuracies_from_list_of_file_specific_column


if __name__ == "__main__":

    cheat_validation = False
    data_version = 10
    epoch_unsup = 400

    for model_name in ["CNP","ConvCNP","UNetCNP"]:
        if model_name in ["ConvCNP","UNetCNP"]:
            pooling = "average"
        else:
            pooling = ""

        if model_name in ["ConvCNP"]:
            layers = [-1]
        elif model_name in ["UNetCNP"]:
            layers = [3,4,5]
        else: layers = [None]

        path_to_base_dir = "../../"

        for layer in layers:

            if model_name in ["ConvCNP","UNetCNP"]:
                plot_dir = "../figures/write_up/semantics/semantics_" + model_name + "_" + str(layer) + "L.svg"
            else:
                plot_dir = "../figures/write_up/semantics/semantics_" + model_name + ".svg"

            list_diff = []
            dirs = {}

            for data_version in range(10,20):
                for i, classifier_name in enumerate(["KNN", "LR", "SVM"]):
                    a = []
                    for semantics in [False,True]:
                        a.append(path_to_base_dir + "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + \
                                ("_cheat_validation" if cheat_validation else "") + "/accuracies/" \
                                + str(data_version) + "V" + "/" + classifier_name + "_on_r_" + model_name + "_" + pooling \
                                + "_" + str(epoch_unsup) + "E" + ".txt")
                    dirs[classifier_name] = a

                if model_name == "CNP":
                    accuracies, list_num_samples = extract_accuracies_from_list_of_files(dirs[classifier_name])
                elif model_name in ["ConvCNP", "UNetCNP"]:
                    accuracies, list_num_samples = extract_accuracies_from_list_of_file_specific_column(dirs[classifier_name],layer)
                diff = [accuracies[1][i] - accuracies[0][i] for i in range(len(accuracies[0]))]
                list_diff.extend(diff)

            max_abs_diff = max(abs(np.array(diff)))
            plt.figure()
            plt.hist(list_diff,50,range=(-0.5 , 0.5))
            plt.xlabel("Difference in accuracy",fontsize=15)
            plt.ylabel("Counts",fontsize=15)
            plt.ylim([0,50])
            plt.savefig(plot_dir)
            plt.close()

