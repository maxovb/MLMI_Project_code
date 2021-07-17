import os
import numpy as np
import random
import torch
from CNPs.OnTheGridConvCNP import ConvCNPExtractRepresentation
from Utils.simple_models import KNN_classifier, LR_classifier, SVM_classifier
from Utils.data_processor import image_processor
from Utils.data_loader import load_supervised_data_as_generator
from Utils.model_loader import load_unsupervised_model

def transform_data_to_representation(model,list_generators, convolutional):
    output = []
    for i,generator in enumerate(list_generators):
        num_samples = len(generator.dataset)
        it = -1
        start = 0
        for data,label in generator:
            it += 1
            if not convolutional:
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                                           convolutional=False, semantic_blocks=None,
                                                                           device=device)
                r = model(x_context,y_context)[:,0,:]
            else:
                mask, context_img = image_processor(data, num_context_points=784,convolutional=convolutional,
                                                    semantic_blocks=None,device=device)
                r = model(mask, context_img)

            n, d = r.shape
            if it == 0:
                X = np.zeros((num_samples, d))
                y = np.zeros(num_samples)

            X[start:start+n,:] = r.cpu().detach().numpy()
            y[start:start+n] = label.cpu().detach().numpy()

            start = start + n
        output.extend([X,y])
    return output

def check_file_not_existent_and_initalize_with_number_of_samples(accuracies_dir_txt,num_training_samples):
    check_file_not_existent(accuracies_dir_txt) 

    # create directory if it doesn't exist yet
    dir_to_create = os.path.dirname(accuracies_dir_txt)
    os.makedirs(dir_to_create, exist_ok=True)

    # initialize the accuracy file with a line showing the size of the training samples
    txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
    with open(accuracies_dir_txt, 'w') as f:
        f.write(txt)

def check_file_not_existent(accuracies_dir_txt):
    assert not (os.path.isfile(accuracies_dir_txt)), "The corresponding accuracies file already exists, please" \
                                                         " remove it to evaluate the models: " + accuracies_dir_txt

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
    batch_size = 64

    random.seed(1234)

    for data_version in range(10):
        # create the model
        model_name = "CNP" # now in for loop
        pooling = "" # now in for loop
        semantics = True
        cheat_validation = False
        for model_name in ["CNP","ConvCNP","UNetCNP"]:
            
            epoch_unsup = 400

            pooling = ""#"flatten"
            model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)

            # get the number of layers possible to investigate
            if model_name == "CNP":
                num_layers = 1
                pooling_types = [""]
            elif model_name == "ConvCNP":
                num_layers = model.CNN.num_residual_blocks
                pooling_types = ["average","flatten"]
            elif model_name in ["UNetCNP","UNetCNP_restrained"]:
                num_layers = 2 * model.CNN.num_down_blocks + 1
                pooling_types = ["average","flatten"]
            else:
                raise "Model name invalid"
            
            for pooling in pooling_types:

                accuracies_dir_txt_knn = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") +\
                                        ("_cheat_validation" if cheat_validation else "") +  "/"\
                                        + str(data_version) + "V" + "/accuracies/KNN_on_r_" + model_name + "_" + pooling + "_" + str(epoch_unsup) + "E" + ".txt"
                accuracies_dir_txt_lr = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") +\
                                        ("_cheat_validation" if cheat_validation else "") +  "/"\
                                        + str(data_version) + "V" + "/accuracies/LR_on_r_" + model_name + "_" + pooling + "_" + str(epoch_unsup) + "E" + ".txt"
                accuracies_dir_txt_svm = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + \
                                        ("_cheat_validation" if cheat_validation else "") + "/"\
                                        + str(data_version) + "V" +"/accuracies/SVM_on_r_" + model_name + "_" + pooling + "_" + str(epoch_unsup) + "E" + ".txt"

                shape_results = (num_layers,len(num_training_samples))
                optimal_k = np.zeros(shape_results)
                optimal_c = np.zeros(shape_results)
                optimal_c_svm = np.zeros(shape_results)
                accuracies_knn = np.zeros(shape_results)
                accuracies_lr = np.zeros(shape_results)
                accuracies_svm = np.zeros(shape_results)

                if os.path.isfile(accuracies_dir_txt_lr):
                    continue

                for layer_id in range(num_layers):

                    print("model name:",model_name,"pooling:",pooling,"layer id:",layer_id)

                    if layer_id == 0:
                        check_file_not_existent(accuracies_dir_txt_knn)
                        check_file_not_existent(accuracies_dir_txt_lr)
                        check_file_not_existent(accuracies_dir_txt_svm)

                    if not(convolutional):
                        model_extract_r = model.encoder.to(device)
                    else:
                        model_extract_r = ConvCNPExtractRepresentation(model,layer_id, pooling=pooling).to(device)

                    for i, num_samples in enumerate(num_training_samples):

                        train_data, valid_data, test_data, img_height, img_widt, num_channels = load_supervised_data_as_generator(batch_size=batch_size, num_training_samples=num_samples, cheat_validation=cheat_validation)

                        X_train, y_train, X_validation, y_validation = transform_data_to_representation(model_extract_r, [train_data, valid_data], convolutional)
                        if num_samples == num_training_samples[0]:
                            X_test, y_test = transform_data_to_representation(model_extract_r, [test_data], convolutional)
                            copy = (X_test, y_test)
                        else:
                            X_test, y_test = copy

                        accuracies_knn[layer_id,i], optimal_k[layer_id,i] = KNN_classifier(X_train,y_train,X_validation,y_validation,X_test,y_test)
                        accuracies_lr[layer_id,i], optimal_c[layer_id,i] = LR_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test)
                        accuracies_svm[layer_id, i], optimal_c_svm[layer_id, i] = SVM_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test)
                
                # create directory if it doesn't exist yet
                dir_to_create = os.path.dirname(accuracies_dir_txt_knn)
                os.makedirs(dir_to_create, exist_ok=True)

                for j in range(len(num_training_samples)):
                    # KNN
                    num_samples = num_training_samples[j]
                    vals = [str(x) for x in accuracies_knn[:,j]]
                    txt_line = str(num_samples) + ", " + " ".join(vals) + "\n"
                    with open(accuracies_dir_txt_knn, 'a+') as f:
                        f.write(txt_line)

                    # LR
                    vals = [str(x) for x in accuracies_lr[:, j]]
                    txt_line = str(num_samples) + ", " + " ".join(vals) + "\n"
                    with open(accuracies_dir_txt_lr, 'a+') as f:
                        f.write(txt_line)

                    # SVM
                    vals = [str(x) for x in accuracies_svm[:, j]]
                    txt_line = str(num_samples) + ", " + " ".join(vals) + "\n"
                    with open(accuracies_dir_txt_svm, 'a+') as f:
                        f.write(txt_line)