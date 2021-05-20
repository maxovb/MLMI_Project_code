import os
import numpy as np
import torch
from CNPs.ConvCNP import ConvCNPExtractRepresentation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
                mask, context_img = image_processor(data, num_context_points=784,convolutional=False,
                                                    semantic_blocks=None,device=device)
                r = model(x_context, y_context)

            n, d = r.shape
            if it == 0:
                X = np.zeros((num_samples, d))
                y = np.zeros(num_samples)

            X[start:start+n,:] = r.cpu().detach().numpy()
            y[start:start+n] = label.cpu().detach().numpy()

            start = start + n
        output.extend([X,y])
    return output

def KNN_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, ks=[1, 2, 3, 5, 7, 10]):
    max_validation_accuracy_knn = 0
    optimal_k = 1
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        validation_accuracy = knn.score(X_validation, y_validation)

        if validation_accuracy > max_validation_accuracy_knn:
            optimal_k = k
            max_validation_accuracy_knn = validation_accuracy
    knn = KNeighborsClassifier(n_neighbors=int(optimal_k)).fit(X_train, y_train)
    return knn.score(X_test, y_test), optimal_k

def LR_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, cs=[1e-2,1,1e2,1e4,1e6,1e8,1e10]):
    max_validation_accuracy_lr = 0
    optimal_c = 1
    for c in cs:
        reg = LogisticRegression(C=c, max_iter=max_iter).fit(X_train, y_train)
        validation_accuracy = reg.score(X_validation, y_validation)

        if validation_accuracy > max_validation_accuracy_lr:
            optimal_c = c
            max_validation_accuracy_lr = validation_accuracy
    reg = LogisticRegression(C=optimal_c, max_iter=max_iter).fit(X_train, y_train)
    return reg.score(X_test, y_test), optimal_c

def check_file_not_existent_and_initalize_with_number_of_samples(accuracies_dir_txt,num_training_samples):
    assert not (os.path.isfile(accuracies_dir_txt)), "The corresponding accuracies file already exists, please" \
                                                         " remove it to evaluate the models: " + accuracies_dir_txt

    # create directory if it doesn't exist yet
    dir_to_create = os.path.dirname(accuracies_dir_txt)
    os.makedirs(dir_to_create, exist_ok=True)

    # initialize the accuracy file with a line showing the size of the training samples
    txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
    with open(accuracies_dir_txt, 'w') as f:
        f.write(txt)

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]

    # create the model
    model_name = "CNP"
    epoch_unsup = 400
    semantics = False
    cheat_validation = False
    pooling = "average"
    model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)

    accuracies_dir_txt_knn = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") +\
                             "/accuracies/KNN_on_r_" + model_name + "_" + str(epoch_unsup) + "E" + ".txt"
    accuracies_dir_txt_lr = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") +\
                            "/accuracies/LR_on_r_" + model_name + "_" + str(epoch_unsup) + "E" + ".txt"

    # get the number of layers possible to investigate
    if model_name == "CNP":
        num_layers = 1
    elif model_name == "ConvCNP":
        num_layers = model.CNN.num_residual_blocks
    elif model_name == "UNetCNP":
        num_layers = 2 * model.CNN.num_down_blocks + 1
    else:
        raise "Model name invalid, was given: " + str(model_name)

    shape_results = (num_layers,len(num_training_samples))
    optimal_k = np.zeros(shape_results)
    optimal_c = np.zeros(shape_results)
    accuracies_knn = np.zeros(shape_results)
    accuracies_lr = np.zeros(shape_results)

    for layer_id in range(num_layers):

        if layer_id = 0:
            check_file_not_existent_and_initalize_with_number_of_samples(accuracies_dir_txt_knn,num_training_samples)
            check_file_not_existent_and_initalize_with_number_of_samples(accuracies_dir_txt_lr, num_training_samples)

        if not(convolutional):
            model_extract_r = model.encoder.to(device)
        else:
            model_extract_r =  ConvCNPExtractRepresentation(model,layer_id, pooling=pooling).to(device)


        for i, num_samples in enumerate(num_training_samples):

            train_data, valid_data, test_data, img_height, img_width = load_supervised_data_as_generator(batch_size=64, num_training_samples=num_samples, cheat_validation=cheat_validation)

            X_train, y_train, X_validation, y_validation = transform_data_to_representation(model_extract_r, [train_data, valid_data], convolutional)
            if num_samples == num_training_samples[0]:
                X_test, y_test = transform_data_to_representation(model_extract_r, [test_data], convolutional)
                copy = (X_test, y_test)
            else:
                X_test, y_test = copy

            accuracies_knn[layer_id,i], optimal_k[i] = KNN_classifier(X_train,y_train,X_validation,y_validation,X_test,y_test)
            accuracies_lr[layer_id,i], optimal_c[i] = LR_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test)

    for j in range(len(num_training_samples)):
        # KNN
        vals = [str(x) for x in accuracies_knn[:,j]]
        txt_line = " ".join(vals) + "\n"
        with open(accuracies_dir_txt_knn, 'a+') as f:
            f.write(txt_line)

        # LR
        vals = [str(x) for x in accuracies_lr[:, j]]
        txt_line = " ".join(vals) + "\n"
        with open(accuracies_dir_txt_lr, 'a+') as f:
            f.write(txt_line)








"""

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the model
    model_name = "CNP"
    epoch_unsup = 400
    semantics = False
    CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
    encoder = CNP_model.encoder.to(device)

    accuracies_dir_txt_knn = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + "/accuracies/KNN_on_r_" + model_name + "_" + str(epoch_unsup) + "E" + ".txt"
    accuracies_dir_txt_lr = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + "/accuracies/LR_on_r_" + model_name + "_" + str(epoch_unsup) + "E" + ".txt"


    ks = [1, 2, 3, 5, 7, 10]  # ,20]
    cs = [1e-2,1,1e2,1e4,1e6,1e8,1e10]
    # TODO: increase max_iter back to 1000
    max_iter = 100
    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
    optimal_k = np.zeros(len(num_training_samples))
    optimal_c = np.zeros(len(num_training_samples))
    accuracies_knn = np.zeros(len(num_training_samples))
    accuracies_lr = np.zeros(len(num_training_samples))

    first_it = True

    for i, num_samples in enumerate(num_training_samples):
        # assess if the accuracies directory already exists, if not create it
        if i == 0:
            assert not(os.path.isfile(accuracies_dir_txt_knn)), "The corresponding accuracies file already exists, please remove it to evaluate the models: " + accuracies_dir_txt_knn
            assert not(os.path.isfile(accuracies_dir_txt_lr)), "The corresponding accuracies file already exists, please remove it to evaluate the models: " + accuracies_dir_txt_lr

            # create directories for the accuracy if they don't exist yet
            dir_to_create = os.path.dirname(accuracies_dir_txt_knn)
            os.makedirs(dir_to_create,exist_ok=True)
            dir_to_create = os.path.dirname(accuracies_dir_txt_lr)
            os.makedirs(dir_to_create,exist_ok=True)

            # initialize the accuracy file with a line showing the size of the training samples
            txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
            with open(accuracies_dir_txt_knn,'w') as f:
                f.write(txt)
            with open(accuracies_dir_txt_lr,'w') as f:
                f.write(txt)


        X_train, y_train, X_validation, y_validation, X_test, y_test = load_supervised_data_as_matrix(num_samples)

        new_X_train = np.zeros((X_train.shape[0], 128))
        new_X_validation = np.zeros((X_validation.shape[0], 128))
        new_X_test = np.zeros((X_test.shape[0], 128))
        X = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
        Xv = np.reshape(X_validation, (X_validation.shape[0], 1, 28, 28))
        Xt = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

        # train data
        data = torch.from_numpy(X)
        x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                   convolutional=False, semantic_blocks=None,
                                                   device=device)
        new_X_train = torch.squeeze(encoder(x_context, y_context)[:, 0]).cpu().detach().numpy()
        X_train = new_X_train

        # validation data
        data = torch.from_numpy(Xv)
        x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                   convolutional=False, semantic_blocks=None,
                                                   device=device)
        new_X_validation = torch.squeeze(encoder(x_context, y_context)[:, 0]).cpu().detach().numpy()
        X_validation = new_X_validation

        # test data
        if first_it:
            first_it = False
            data = torch.from_numpy(Xt)
            x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                          convolutional=False, semantic_blocks=None,
                                                           device=device)

            for n in range(x_context.shape[0]//1000):
                new_X_test[n*1000:(n+1)*1000] = torch.squeeze(encoder(x_context[n*1000:(n+1)*1000], y_context[n*1000:(n+1)*1000])[:, 0]).cpu().detach().numpy()

            X_test = new_X_test

            #TODO: remove this 
            #X_test = X_test[:500]
            #y_test = y_test[:500]


            cop = (X_test,y_test)
        else:
            X_test,y_test = cop

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


"""