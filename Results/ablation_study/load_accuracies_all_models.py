import os.path

from accuracy_plots import extract_accuracies_from_list_of_files

def load_accuracies_all_models(base_dir,model_name,model_size,percentage_unlabelled_set,data_versions=list(range(1,11)),semantics=True,pooling="average",layer_id=4,gamma=1.5):
    all_accuracies = {}
    for classify_same_image in [False, True]:
        for consistency_regularization in [False,True]:
            for grad_norm in [False,True]:
                list_acc_files = []
                for data_version in data_versions:
                    accuracies_dir_txt = base_dir + "saved_models/MNIST/joint" + ("_semantics" if semantics else "")\
                                         + ("_cons" if consistency_regularization else "") \
                                         + ("_GN_" + str(gamma) + "" if grad_norm else "") \
                                         + ("_ET/" if classify_same_image else "/") \
                                         + "accuracies/" + str(percentage_unlabelled_set) + "P_" \
                                         + str(data_version) + "V/" + model_name + "_" + model_size \
                                         + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") \
                                         + ".txt"

                    if os.path.isfile(accuracies_dir_txt):
                        list_acc_files.append(accuracies_dir_txt)

                local_accuracies, list_num_samples = extract_accuracies_from_list_of_files(list_acc_files)
                local_accuracies = [x[0] for x in local_accuracies]
                key = model_name + ("_CL" if consistency_regularization else "") \
                      + ("_ET" if classify_same_image else "") \
                      + ("_GN" if grad_norm else "")
                all_accuracies[key] = local_accuracies
    return all_accuracies






