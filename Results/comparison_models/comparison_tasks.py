import os
import sys
sys.path.insert(1, "../../Utils")
from helper_results import plot_loss

if __name__ == "__main__":
    path_to_base_dir = "../../"
    plot_base_dir = path_to_base_dir + "Results/figures/comparisons_tasks/"
    semantics = True
    gamma = 1.5
    percentage_unlabelled_set = 0.25
    data_version = 0
    num_samples = 100


    for model_name in ["UNetCNP","UNetCNP_GMM"]:

        labels = [model_name, "+ consistency", "+ extra task", "+ both"]
        styles = ["-","--","-.",":"]
        for grad_norm in [False, True]:
            loss_base_dir = ["", "", "", ""]
            i = -1
            for classify_same_image in [False, True]:
                for consistency_regularization in [False, True]:
                    i += 1
                    experiment_dir_list = [path_to_base_dir + "saved_models/MNIST/joint"
                                           + ("_semantics" if semantics else "_")
                                           + ("_cons" if consistency_regularization else "")
                                           + ("_GN_" + str(gamma) + "" if grad_norm else "")
                                           + ("_ET/" if classify_same_image else "/")
                                           + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/"
                                           + str(num_samples) + "S/", model_name, "/"]
                    experiment_dir_txt = "".join(experiment_dir_list)
                    loss_base_dir[i] = experiment_dir_txt + "loss/"

            plot_dir_list = [plot_base_dir + "MNIST/joint" + ("_semantics" if semantics else "_")
                             + ("_GN_" + str(gamma) + "/" if grad_norm else "/")
                             + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/"
                             + str(num_samples) + "S/", model_name, "/"]
            plots_dir = "".join(plot_dir_list)

            dir_to_create = os.path.dirname("".join(plots_dir))
            os.makedirs(dir_to_create, exist_ok=True)

            for file_name in os.listdir(loss_base_dir[0]):
                if file_name[-4:] == ".txt":
                    list_files = [loss_base_dir[i] + file_name for i in range(len(loss_base_dir))]
                    if "loss" in file_name:
                        y_label = "Loss"
                    elif "accuracy" in file_name:
                        y_label = "Accuracy"
                    elif "total" in file_name:
                        y_label = "Total"
                    plot_current_dir = plots_dir + file_name[:-4] + ".svg"
                    plot_loss(list_files,plot_current_dir,labels,y_label,styles)
