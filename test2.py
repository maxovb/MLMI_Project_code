import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from Utils.data_processor import image_processor
from Utils.data_loader import load_supervised_data_as_matrix
from Utils.model_loader import load_unsupervised_model


if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the model
    model_name = "CNP"
    epoch_unsup = 400
    semantics = False
    cheat_validation = False
    CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
    encoder = CNP_model.encoder


    ks = [1, 2, 3, 5, 7, 10]  # ,20]
    cs = [1e-2,1,1e2,1e4,1e6,1e8,1e10]
    max_iter = 1000
    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
    optimal_k = np.zeros(len(num_training_samples))
    optimal_c = np.zeros(len(num_training_samples))
    accuracies_knn = np.zeros(len(num_training_samples))
    accuracies_lr = np.zeros(len(num_training_samples))

    first_it = True

    for i, num_samples in enumerate(num_training_samples):

        X_train, y_train, X_validation, y_validation, X_test, y_test = load_supervised_data_as_matrix(num_samples)

        #TODO: remove this
        X_test = X_test[:500]
        y_test = y_test[:500]

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
        new_X_train = torch.squeeze(encoder(x_context, y_context)[:, 0]).detach().numpy()
        X_train = new_X_train

        # validation data
        data = torch.from_numpy(Xv)
        x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                   convolutional=False, semantic_blocks=None,
                                                   device=device)
        new_X_validation = torch.squeeze(encoder(x_context, y_context)[:, 0]).detach().numpy()
        X_validation = new_X_validation

        # test data
        if first_it:
            first_it = False
            data = torch.from_numpy(Xt)
            x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                          convolutional=False, semantic_blocks=None,
                                                           device=device)
            new_X_test = torch.squeeze(encoder(x_context, y_context)[:, 0]).detach().numpy()
            X_test = new_X_test
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

    print("KNN")
    for i, num_samples in enumerate(num_training_samples):
        print("KNN num_samples", num_samples, "accuracy:", accuracies_knn[i], "optimal k:", optimal_k[i])

    print("LR")
    for i, num_samples in enumerate(num_training_samples):
        print("LR num_samples", num_samples, "accuracy:", accuracies_lr[i], "optimal c:", optimal_c[i])


