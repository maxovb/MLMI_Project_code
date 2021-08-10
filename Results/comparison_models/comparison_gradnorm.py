import os
import sys
import matplotlib.pyplot as plt
import math
sys.path.insert(1, "../../Utils")
from helper_results import plot_loss

if __name__ == "__main__":
    path_to_base_dir = "../../"
    plot_base_dir = path_to_base_dir + "Results/figures/comparisons_with_gradnorm/"
    semantics = True
    gamma = 1.5
    percentage_unlabelled_set = 0.0625
    data_version = 0
    num_samples = 100

    labels = ["Without GradNorm","With GradNorm"]

    for model_name in ["UNetCNP"]:#,"UNetCNP_GMM"]:
        for consistency_regularization in [False,True]:
            for classify_same_image in [False,True]:
                plot_dir_list = [plot_base_dir + "MNIST/joint" + ("_semantics" if semantics else "_")
                                 + ("_cons" if consistency_regularization else "")
                                 + ("_ET/" if classify_same_image else "/")
                                 + str(percentage_unlabelled_set) + "P_"
                                 + str(num_samples) + "S/", model_name, "/"]
                plots_dir = "".join(plot_dir_list)

                dir_to_create = os.path.dirname("".join(plots_dir))
                os.makedirs(dir_to_create, exist_ok=True)
                loss_base_dir = []

                for i, grad_norm in enumerate([False, True]):
                    for data_version in range(1,11):
                        experiment_dir_list = [path_to_base_dir + "saved_models/MNIST/joint"
                                               + ("_semantics" if semantics else "_")
                                               + ("_cons" if consistency_regularization else "")
                                               + ("_GN_" + str(gamma) + "" if grad_norm else "")
                                               + ("_ET/" if classify_same_image else "/")
                                               + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/"
                                               + str(num_samples) + "S/", model_name, "/"]
                        experiment_dir_txt = "".join(experiment_dir_list)
                        loss_base_dir.append(experiment_dir_txt + "loss/")

                num_cols = 4
                num_plots = 2 + (1 if  consistency_regularization else 0) + (1 if classify_same_image else 0)
                num_rows = math.ceil(num_plots / num_cols)
                found = 0

                plot_overview_dir = plots_dir + "overview" + ("_cons" if consistency_regularization else "") + ("_ET" if classify_same_image else "") + ".svg"
                #fig, ax = plt.subplots(num_rows, min(num_cols, num_plots), figsize=(num_cols * 5, num_rows * 5))
                fig, ax = plt.subplots(num_rows, 4, figsize=(num_cols * 5, num_rows * 5))
                if num_rows != 1 or num_plots != 1:
                    if num_rows == 1:
                        ax = ax[None, :]
                    elif num_cols == 1:
                        ax = ax[:, None]

                current_col = -1
                current_row = 0

                for file_name in os.listdir(loss_base_dir[0]):
                    overview_position = None
                    ylim = None

                    if file_name[-4:] == ".txt":
                        list_files = [loss_base_dir[i] + file_name for i in range(len(loss_base_dir))]
                        if "loss" in file_name:
                            y_label = "Loss"
                        if "cons" in file_name:
                            y_label = "Consistency loss"
                            overview_position = (-1 if not(classify_same_image) else -2)
                        if "rec" in file_name:
                            y_label = "Reconstruction loss"
                            overview_position = 0
                            ylim = [-1200,-200]
                        if "accuracy" in file_name:
                            y_label = "Accuracy"
                            overview_position = 1
                        if "accuracy_discr" in file_name:
                            y_label = "Siamese Accuracy"
                            overview_position = -1
                        elif "total" in file_name:
                            y_label = "Total"
                        plot_current_dir = plots_dir + file_name[:-4] + ".svg"
                        plot_loss(list_files,plot_current_dir,labels,y_label, ylim = ylim, running_avg=True,groups=10)

                        if overview_position != None:
                            found += 1
                            labels = ["Without GradNorm", "With GradNorm"]
                            if "validation" in file_name:
                                labels = "valid also show dashed"
                            plot_loss(list_files, plot_overview_dir, labels, y_label, ylim = ylim, ax=ax[0,overview_position], running_avg=True, groups=10)
                            if found == 2 * num_plots:
                                plt.tight_layout()
                                if num_plots < 3:
                                    ax[0,2].set_axis_off()
                                if num_plots < 4:
                                    ax[0,3].set_axis_off()
                                fig.savefig(plot_overview_dir) #since axes are given the above function doesn't save it3

