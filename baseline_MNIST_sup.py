import torch
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from Networks.create_lenet import create_lenet
from Train.train_CNP_images import train_sup
from Utils.simple_models import KNN_classifier, LR_classifier, SVM_classifier
from Utils.data_loader import load_supervised_data_as_matrix,load_supervised_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss

if __name__ == "__main__":
    random.seed(1234)

    """
    #####  K-nearest neighbour #####
    print("KNN, LR and SVM")

    accuracies_dir_txt_knn = "saved_models/MNIST/supervised/accuracies/KNN.txt"
    accuracies_dir_txt_lr = "saved_models/MNIST/supervised/accuracies/LR.txt"
    accuracies_dir_txt_SVM = "saved_models/MNIST/supervised/accuracies/SVM.txt"

    ks = [1,2,3,5,7,10]
    cs = [1e-2,1,1e2,1e4,1e6,1e8,1e10]
    max_iter = 1000
    num_training_samples = [10, 20, 40, 60, 80, 100, 600, 1000, 3000]
    optimal_k = np.zeros(len(num_training_samples))
    accuracies_knn = np.zeros(len(num_training_samples))
    optimal_c = np.zeros(len(num_training_samples))
    accuracies_lr = np.zeros(len(num_training_samples))
    optimal_c_SVM = np.zeros(len(num_training_samples))
    accuracies_SVM = np.zeros(len(num_training_samples))
    
    for i,num_samples in enumerate(num_training_samples):
        if i == 0:  # at the iteration over the different number of training samples
            # create directories for the accuracy if they don't exist yet
            dir_to_create = os.path.dirname(accuracies_dir_txt_knn)
            os.makedirs(dir_to_create, exist_ok=True)
            dir_to_create = os.path.dirname(accuracies_dir_txt_lr)
            os.makedirs(dir_to_create, exist_ok=True)
            dir_to_create = os.path.dirname(accuracies_dir_txt_SVM)
            os.makedirs(dir_to_create, exist_ok=True)

        X_train, y_train, X_validation, y_validation, X_test, y_test = load_supervised_data_as_matrix(num_samples)
        accuracies_knn[i], optimal_k[i] = KNN_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, ks=ks)
        accuracies_lr[i], optimal_c[i] = LR_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, cs=cs, max_iter=max_iter)
        accuracies_SVM[i], optimal_c_SVM[i] = SVM_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, cs=cs, max_iter=max_iter)

        # write the accuracy to the text file
        with open(accuracies_dir_txt_knn, 'a+') as f:
            text = str(num_samples) +", " + str(accuracies_knn[i]) + "\n"
            f.write(text)
        with open(accuracies_dir_txt_lr, 'a+') as f:
            text = str(num_samples) +", " + str(accuracies_lr[i]) + "\n"
            f.write(text)
        with open(accuracies_dir_txt_SVM, 'a+') as f:
            text = str(num_samples) +", " + str(accuracies_SVM[i]) + "\n"
            f.write(text)
    """

    ##### LeNet #####
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    dropout = True
    model_name = "LeNet" + ("_dropout" if dropout else "")  # one of ["CNP", "ConvCNP", "ConvCNPXL"]
    print(model_name)

    cheat_validation= False # use a large validation set even if the trainign data is small

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

            if num_samples <= 200:
                batch_size = 64
                learning_rate = 5e-3
                epochs = 200
                save_freq = 20
            else:
                batch_size = 64
                learning_rate = 1e-3
                epochs = 200
                save_freq = 20

            # load the supervised set
            train_data, validation_data, test_data, img_height, img_width, num_channels = load_supervised_data_as_generator(batch_size, num_samples,cheat_validation=cheat_validation)

            # create the model
            model = create_lenet(model_size,dropout)
            model.to(device)

            # define the directories
            model_save_dir = ["saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/", model_name, "/", model_name, "_",model_size, "", "E", ".pth"]
            train_loss_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + "_train.txt"
            validation_loss_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size +  "_validation.txt"
            loss_dir_plot = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/"+ model_name + "/loss/" + model_name + "_" + model_size + ".svg"
            accuracies_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + "accuracies/" + model_name + "_" + model_size + ".txt"

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

                num_context_points = 28 * 28
                accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=False,num_context_points=num_context_points, save_freq=save_freq, is_CNP=False)

                # write the accuracy to the text file
                with open(accuracies_dir_txt, 'a+') as f:
                    text = str(num_samples) +", " + str(accuracy)
                    f.write(text)
