import os
import numpy as np
import torch
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from Utils.data_processor import image_processor
from Utils.data_loader import load_supervised_data_as_matrix
from Utils.model_loader import load_unsupervised_model

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the model
    model_name = "UNetCNP"
    epoch_unsup = 250
    semantics = True

    CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
    num_down_blocks = CNP_model.CNN.num_down_blocks
    CNP_model = CNP_model.to(device)

    step = 200

    for layer_id in range(2*num_down_blocks):
        print("Layer id:", layer_id)

        CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
        num_down_blocks = CNP_model.CNN.num_down_blocks
        CNP_model = CNP_model.to(device)

        accuracies_dir_txt_knn = "saved_models/MNIST/supervised" + (
            "_semantics" if semantics else "") + "/accuracies/KNN_on_r_" + model_name + "_" + str(
            epoch_unsup) + "E" + "_layer_" + str(layer_id) + ".txt"
        accuracies_dir_txt_lr = "saved_models/MNIST/supervised" + (
            "_semantics" if semantics else "") + "/accuracies/LR_on_r_" + model_name + "_" + str(
            epoch_unsup) + "E" + "_layer_" + str(layer_id) + ".txt"

        feature_map = {}
        feature_map["current"] = torch.ones((1,1,1,1,1))
        def hook_fn(m, i, o):
            feature_map["current"] = o
        if layer_id < num_down_blocks:
            CNP_model.CNN.h_down[layer_id].register_forward_hook(hook_fn)
        else:
            CNP_model.CNN.h_up[layer_id-num_down_blocks].register_forward_hook(hook_fn)

        ks = [1, 2, 3, 5, 7, 10]  # ,20]
        cs = [1e-2, 1, 1e2, 1e4, 1e6, 1e8, 1e10]
        # TODO: increase max_iter back to 1000
        max_iter = 100
        num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
        optimal_k = np.zeros(len(num_training_samples))
        optimal_c = np.zeros(len(num_training_samples))
        accuracies_knn = np.zeros(len(num_training_samples))
        accuracies_lr = np.zeros(len(num_training_samples))

        first_it = True

        for i, num_samples in enumerate(num_training_samples):
            # assess if the accuracies directory already exists, if not create it
            if i == 0:
                assert not (os.path.isfile(
                    accuracies_dir_txt_knn)), "The corresponding accuracies file already exists, please remove it to evaluate the models: " + accuracies_dir_txt_knn
                assert not (os.path.isfile(
                    accuracies_dir_txt_lr)), "The corresponding accuracies file already exists, please remove it to evaluate the models: " + accuracies_dir_txt_lr

                # create directories for the accuracy if they don't exist yet
                dir_to_create = os.path.dirname(accuracies_dir_txt_knn)
                os.makedirs(dir_to_create, exist_ok=True)
                dir_to_create = os.path.dirname(accuracies_dir_txt_lr)
                os.makedirs(dir_to_create, exist_ok=True)

                # initialize the accuracy file with a line showing the size of the training samples
                txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
                with open(accuracies_dir_txt_knn, 'w') as f:
                    f.write(txt)
                with open(accuracies_dir_txt_lr, 'w') as f:
                    f.write(txt)

            X_train, y_train, X_validation, y_validation, X_test, y_test = load_supervised_data_as_matrix(num_samples)

            X = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
            Xv = np.reshape(X_validation, (X_validation.shape[0], 1, 28, 28))
            Xt = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

            # train data
            data = torch.from_numpy(X)
            if not(convolutional):
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                                           convolutional=convolutional, semantic_blocks=None,
                                                                           device=device)
                out = CNP_model(x_context, y_context)
                x = torch.mean(feature_map["current"], dim=[-2, -1])
                X_train = x.cpu().detach().numpy()
            else:
                mask, context_img = image_processor(data, num_context_points=784,
                                                                           convolutional=convolutional,
                                                                           semantic_blocks=None,
                                                                           device=device)
                for n in range(math.ceil(mask.shape[0] / step)):
                    out = CNP_model(mask[n * step:(n + 1) * step], context_img[n * step:(n + 1) * step])
                    x = torch.mean(feature_map["current"], dim=[-2, -1])
                    if n == 0:
                        c = x.cpu().detach().numpy().shape[1]
                        new_X_train = np.zeros((X.shape[0], c))
                    new_X_train[n * step:(n + 1) * step] = x.cpu().detach().numpy()
                X_train = new_X_train

            # validation data
            data = torch.from_numpy(Xv)
            if not (convolutional):
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                                           convolutional=convolutional,
                                                                           semantic_blocks=None,
                                                                           device=device)
                out = CNP_model(x_context, y_context)
                x = torch.mean(feature_map["current"], dim=[-2, -1])
                X_validation = x.cpu().detach().numpy()

            else:
                mask, context_img = image_processor(data, num_context_points=784,
                                                    convolutional=convolutional,
                                                    semantic_blocks=None,
                                                    device=device)
                for n in range(math.ceil(mask.shape[0] / step)):
                    out = CNP_model(mask[n * step:(n + 1) * step], context_img[n * step:(n + 1) * step])
                    x = torch.mean(feature_map["current"], dim=[-2, -1])
                    if n == 0:
                        c = x.cpu().detach().numpy().shape[1]
                        new_X_validation = np.zeros((Xv.shape[0], c))
                    new_X_validation[n * step:(n + 1) * step] = x.cpu().detach().numpy()
                X_validation = new_X_validation

            # test data
            if first_it:
                first_it = False
                data = torch.from_numpy(Xt)
                if not (convolutional):
                    x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                                               convolutional=convolutional,
                                                                               semantic_blocks=None,
                                                                               device=device)
                    for n in range(math.ceil(x_context.shape[0] / step)):
                        out = CNP_model(x_context[n * step:(n + 1) * step], y_context[n * step:(n + 1) * step])
                        x = torch.mean(feature_map["current"], dim=[-2, -1])
                        if n == 0:
                            c = x.cpu().detach().numpy().shape[1]
                            new_X_test = np.zeros((Xt.shape[0],c))
                        new_X_test[n * step:(n + 1) * step] = x.cpu().detach().numpy()
                else:
                    mask, context_img = image_processor(data, num_context_points=784,
                                                                               convolutional=convolutional,
                                                                               semantic_blocks=None,
                                                                               device=device)
                    for n in range(math.ceil(mask.shape[0] / step)):
                        out = CNP_model(mask[n * step:(n + 1) * step], context_img[n * step:(n + 1) * step])
                        x = torch.mean(feature_map["current"], dim=[-2, -1])
                        if n == 0:
                            c = x.cpu().detach().numpy().shape[1]
                            new_X_test = np.zeros((Xt.shape[0], c))
                        new_X_test[n * step:(n + 1) * step] = x.cpu().detach().numpy()

                X_test = new_X_test

                # TODO: remove this (debugging stuff)
                X_test = X_test[:500]
                y_test = y_test[:500]

                cop = (X_test, y_test)
            else:
                X_test, y_test = cop

            if len(X_validation.shape) == 1:
                X_validation = X_validation[np.newaxis, :]

            max_validation_accuracy_knn = 0

            for k in ks:
                knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                validation_accuracy = knn.score(X_validation, y_validation)

                if validation_accuracy > max_validation_accuracy_knn:
                    optimal_k[i] = k
                    max_validation_accuracy_knn = validation_accuracy
            knn = KNeighborsClassifier(n_neighbors=int(optimal_k[i])).fit(X_train, y_train)
            accuracies_knn[i] = knn.score(X_test, y_test)

            max_validation_accuracy_lr = 0

            for c in cs:
                reg = LogisticRegression(C=c, max_iter=max_iter).fit(X_train, y_train)
                validation_accuracy = reg.score(X_validation, y_validation)

                if validation_accuracy > max_validation_accuracy_lr:
                    optimal_c[i] = c
                    max_validation_accuracy_lr = validation_accuracy
            reg = LogisticRegression(C=optimal_c[i], max_iter=max_iter).fit(X_train, y_train)
            accuracies_lr[i] = reg.score(X_test, y_test)

            # write the accuracies to the text file
            with open(accuracies_dir_txt_knn, 'a+') as f:
                f.write('%s\n' % accuracies_knn[i])
            with open(accuracies_dir_txt_lr, 'a+') as f:
                f.write('%s\n' % accuracies_lr[i])

        print("KNN")
        for i, num_samples in enumerate(num_training_samples):
            print("KNN num_samples", num_samples, "accuracy:", accuracies_knn[i], "optimal k:", optimal_k[i])

        print("LR")
        for i, num_samples in enumerate(num_training_samples):
            print("LR num_samples", num_samples, "accuracy:", accuracies_lr[i], "optimal c:", optimal_c[i])