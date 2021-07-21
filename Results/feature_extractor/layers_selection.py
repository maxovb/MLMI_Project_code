import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1,"../")
from accuracy_plots import extract_accuracies_form_file_with_multiple_columns, plot_accuracy_vs_layer_num


if __name__ == "__main__":
    cheat_validation = False
    epoch_unsup = 400

    path_to_base_dir = "../../"

    for semantics in [False,True]:
        for model_name in ["ConvCNP","UNetCNP"]:
            acc_dir_plot = "../figures/write_up/feature_extractor/layer_selection/mean_acc_vs_layer_"\
                           + ("semantics_" if semantics else "_") + model_name + ".svg"

            if model_name in ["ConvCNP","UNetCNP"]:
                pooling = "average"
            else:
                pooling = ""

            for classifier_name in ["KNN","LR","SVM"]:

                ls = []
                for data_version in range(10,20):
                    dir_txt = (path_to_base_dir + "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + \
                             ("_cheat_validation" if cheat_validation else "") + "/accuracies/" \
                             + str(data_version) + "V" + "/" + classifier_name + "_on_r_" + model_name + "_" + pooling \
                             + "_" + str(epoch_unsup) + "E" + ".txt")

                    accuracies, list_num_samples = extract_accuracies_form_file_with_multiple_columns(dir_txt)
                    ls.append(accuracies)

            accuracies_array = np.stack(ls,axis=2)
            accuracies_array_mean = np.mean(accuracies_array,axis=2)
            accuracies_array_mean = accuracies_array_mean.T
            accuracies_mean_transposed = accuracies_array_mean.tolist()
            labels = []
            for x in list_num_samples:
                labels.append(str(int(x)) + " samples")
            plot_accuracy_vs_layer_num(accuracies_mean_transposed, list(range(len(accuracies))), acc_dir_plot, labels)