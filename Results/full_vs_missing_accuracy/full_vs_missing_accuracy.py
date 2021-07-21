import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    model_name = "UNetCNP"
    model_size = "medium_dropout"
    semantics = True
    consistency_regularization = True
    classify_same_image = False
    grad_norm = True
    gamma = 1.5
    R = 1.0
    percentage_unlabelled_set = 0.25/4
    data_version = 0
    num_samples = 100
    pooling = "average"
    layer_id = 4

    path_to_base_dir = "../../"

    experiment_dir_list = [path_to_base_dir + "saved_models/MNIST/joint_" + str(R) + "R" + ("_semantics" if semantics else "_") + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") + ("_ET/" if classify_same_image else "/") + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/" + str(num_samples) + "S/", model_name, "/"]
    experiment_dir_txt = "".join(experiment_dir_list)
    validation_losses_dir_list = [experiment_dir_txt + "loss/" + model_name + "_" + model_size +
                                  ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "_")
                                  + "_validation_", "", ".txt"]

    dir_plot = "../figures/full_vs_missing_accuracy/" +  "joint_" + str(R) + "R" + ("_semantics" if semantics else "_")\
               + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") \
               + ("_ET/" if classify_same_image else "/") + str(percentage_unlabelled_set) + "P_" + str(data_version)\
               + "V/" + str(num_samples) + "S/" + model_name + "_" + model_size \
               + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "_") \
               + "_validation.svg"


    filepath = validation_losses_dir_list.copy()
    filepath[1] = "accuracy"
    filepath = "".join(filepath)

    train_acc = []
    with open(filepath,"r") as f:
        for line in f:
            x = float(line.split("\n")[0])
            train_acc.append(x)

    filepath = validation_losses_dir_list.copy()
    filepath[1] = "full_accuracy"
    filepath = "".join(filepath)

    train_full_acc = []
    epochs = []
    with open(filepath, "r") as f:
        for line in f:
            e = float(line.split(",")[0])
            epochs.append(e)
            x = float((line.split(",")[1]).split("\n")[0])
            train_full_acc.append(x)

    z = zip(train_full_acc,epochs)
    sorted_zip_list = sorted(z)
    sorted_train_full_acc = [x for x,_ in sorted(sorted_zip_list)]
    sorted_epochs = sorted(epochs)

    dir_to_create = os.path.dirname(dir_plot)
    os.makedirs(dir_to_create, exist_ok=True)

    plt.figure()
    plt.plot(range(len(train_acc)),train_acc)
    plt.plot(sorted_epochs,sorted_train_full_acc)
    plt.legend(["Images with missing pixels","Full images"],fontsize="x-large")
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.ylim([0,1])
    plt.savefig(dir_plot)



