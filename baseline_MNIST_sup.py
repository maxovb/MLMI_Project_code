import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from Networks.create_lenet import create_lenet
from Train.train_CNP_images import train_sup
from Utils.data_loader import load_supervised_data_as_matrix,load_supervised_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss

if __name__ == "__main__":

    #####  K-nearest neighbour #####
    print("KNN")

    accuracies_dir_txt_knn = "saved_models/MNIST/supervised/accuracies/KNN.txt"

    ks = [1,2,3,5,7,10]#,20]
    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
    optimal_k = np.zeros(len(num_training_samples))
    accuracies = np.zeros(len(num_training_samples))
    for i,num_samples in enumerate(num_training_samples):
        if i == 0:  # at the iteration over the different number of training samples
            # create directories for the accuracy if they don't exist yet
            dir_to_create = os.path.dirname(accuracies_dir_txt_knn)
            os.makedirs(dir_to_create, exist_ok=True)

            # initialize the accuracy file with a line showing the size of the training samples
            txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
            with open(accuracies_dir_txt_knn, 'w') as f:
                f.write(txt)

        X_train, y_train, X_validation, y_validation, X_test, y_test = load_supervised_data_as_matrix(num_samples)
        max_validation_accuracy = 0
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
            validation_accuracy = knn.score(X_validation,y_validation)
            if validation_accuracy > max_validation_accuracy:
                optimal_k[i] = k
                max_validation_accuracy = validation_accuracy
        knn = KNeighborsClassifier(n_neighbors=int(optimal_k[i])).fit(X_train, y_train)
        accuracies[i] = knn.score(X_test,y_test)

        # write the accuracy to the text file
        with open(accuracies_dir_txt_knn, 'a+') as f:
            f.write('%s\n' % accuracies[i])


    ##### LeNet #####

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "LeNet"  # one of ["CNP", "ConvCNP", "ConvCNPXL"]
    print(model_name)

    # for continued supervised training
    train = True
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 20  # which epoch to start from
    else:
        epoch_start = 0


    # training parameters
    num_training_samples = [10,20,40,60,80,100,600,1000,3000]

    for model_size in ["small","medium","large"]:
        for i,num_samples in enumerate(num_training_samples):

            if num_samples <= 60:
                batch_size = 64
                learning_rate = 5e-3
                epochs = 1000
                save_freq = 100
            elif num_samples <= 100:
                batch_size = 64
                learning_rate = 5e-3
                epochs = 400
                save_freq = 40
            else:
                batch_size = 64
                learning_rate = 1e-3
                epochs = 200
                save_freq = 20

            # load the supervised set
            train_data, validation_data, test_data, img_height, img_width = load_supervised_data_as_generator(
                batch_size, num_samples)

            # create the model
            model = create_lenet(model_size)

            # define the directories
            model_save_dir = ["saved_models/MNIST/supervised/" + str(num_samples) + "S/", model_name, "/", model_name, "_",model_size, "", "E", ".pth"]
            train_loss_dir_txt = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + "_train.txt"
            validation_loss_dir_txt = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size +  "_validation.txt"
            loss_dir_plot = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ".svg"
            accuracies_dir_txt = "saved_models/MNIST/supervised/accuracies/" + model_name + "_" + model_size + ".txt"

            # create directories for the checkpoints and loss files if they don't exist yet
            dir_to_create = "".join(model_save_dir[:3]) + "loss/"
            os.makedirs(dir_to_create, exist_ok=True)

            if load:
                load_dir = model_save_dir.copy()
                load_dir[-3] = str(epoch_start)
                load_dir = "".join(load_dir)

                if train:
                    # check if the loss file is valid
                    with open(train_loss_dir_txt, 'r') as f:
                        nbr_losses = len(f.read().split())

                    assert nbr_losses == epoch_start, "The number of lines in the loss file does not correspond to the number of epochs"

                # load the model
                model.load_state_dict(torch.load(load_dir, map_location=device))
            else:
                # if train from scratch, check if a loss file already exists (it should not, so remove it if necessary)
                assert not (os.path.isfile(train_loss_dir_txt)), "The corresponding loss file already exists, please remove it to train from scratch"

            if train:
                avg_loss_per_epoch = train_sup(train_data, model, epochs, model_save_dir, train_loss_dir_txt,
                                                   validation_data=validation_data,
                                                   validation_loss_dir_txt=validation_loss_dir_txt, is_CNP = False,
                                                   save_freq=save_freq,
                                                   epoch_start=epoch_start, device=device, learning_rate=learning_rate)
                plot_loss([train_loss_dir_txt, validation_loss_dir_txt], loss_dir_plot)

            if save:
                save_dir = model_save_dir.copy()
                save_dir[-3] = str(epoch_start + epochs)
                save_dir = "".join(save_dir)
                torch.save(model.state_dict(), save_dir)

            if evaluate:
                if i == 0: # at the iteration over the different number of training samples
                    # create directories for the accuracy if they don't exist yet
                    dir_to_create = os.path.dirname(accuracies_dir_txt)
                    os.makedirs(dir_to_create, exist_ok=True)

                    # initialize the accuracy file with a line showing the size of the training samples
                    txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
                    with open(accuracies_dir_txt, 'w') as f:
                        f.write(txt)

                num_context_points = 28 * 28
                accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=False,num_context_points=num_context_points, save_freq=save_freq, is_CNP=False)

                # write the accuracy to the text file
                with open(accuracies_dir_txt, 'a+') as f:
                    f.write('%s\n' % accuracy)

