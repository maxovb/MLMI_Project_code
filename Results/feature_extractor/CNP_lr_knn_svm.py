import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, "../")
from accuracy_plots import extract_accuracies_from_list_of_files, plot_accuracy


if __name__ == "__main__":
    cheat_validation = False
    epochs_unsup = 400

    labels = ["Random", "Semantics"]
    model_name = "CNP"
    for classifier_type in ["LR","KNN","SVM"]:
        acc_dir_plot = "../figures/write_up/feature_extractor/semantics_comparison/CNP_" \
                       + classifier_type + ".svg"

        styles_r = ["r-", "b--"]
        list_acc_dir_txt = []
        for semantics in [False,True]:
            accuracies_dir_txt = "../../saved_models/MNIST/supervised" \
                                 + ("_semantics" if semantics else "") \
                                 + ("_cheat_validation" if cheat_validation else "")\
                                 + "/accuracies/" + classifier_type + "_on_r_"\
                                 + model_name + "__" + str(epochs_unsup) + "E.txt"
            list_acc_dir_txt.append(accuracies_dir_txt)

        accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_dir_txt)
        plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_r)