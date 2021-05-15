import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchsummary import summary
from Train.train_CNP_images import train_sup
from CNPs.create_model import  create_model
from CNPs.modify_model_for_classification import modify_model_for_classification
from Utils.data_loader import load_supervised_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss


def load_unsupervised_model(model_name, epoch, semantics = False, device = torch.device('cpu')):
    model_load_dir = ["saved_models/MNIST/", model_name + ("_semantics" if semantics else ""), "/", model_name + ("_semantics" if semantics else ""), "_", str(epoch), "E", ".pth"]
    load_dir = "".join(model_load_dir)

    # create the model
    model, convolutional = create_model(model_name)

    # load the checkpoint
    model.load_state_dict(torch.load(load_dir, map_location=device))

    return model,convolutional


if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "CNP" # one of ["CNP", "ConvCNP", "ConvCNPXL"]
    model_size = "large" # one of ["small","medium","large"]

    freeze_weights = True # freeze the weights of the part taken from the unsupervised model
    cheat_validation= True # use a large validation set even if the trainign data is small
    semantics = True # use the ConvCNP and CNP pre-trained with blocks of context pixels, i.e. carry more semantics

    for model_name in ["CNP","ConvCNP"]:
        for model_size in ["small","medium","large"]:
            print(model_name, model_size)

            # for continued supervised training
            train = True
            load = False
            save = False
            evaluate = True
            if load:
                epoch_start = 100 # which epoch to start from
            else:
                epoch_start = 0

            # parameters from the model to load
            epoch_unsup = 400 # unsupervised model to load initially

            # training parameters
            num_training_samples = [10,20,40,60,80,100,600,1000,3000]

            for i,num_samples in enumerate(num_training_samples):
                if num_samples <= 60:
                    batch_size = 64
                    learning_rate = 1e-3
                    epochs = 400
                    save_freq = 20
                elif num_samples <= 100:
                    batch_size = 64
                    learning_rate = 1e-3
                    epochs = 400
                    save_freq = 20
                else:
                    batch_size = 64
                    learning_rate = 1e-3
                    epochs = 200
                    save_freq = 20

                # load the supervised set
                train_data, validation_data, test_data, img_height, img_width = load_supervised_data_as_generator(batch_size, num_samples,cheat_validation=cheat_validation)

                # create the model
                CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)

                # modify the model to act as a classifier
                model = modify_model_for_classification(CNP_model,model_size,convolutional,freeze_weights,img_height=img_height,img_width=img_width)
                model.to(device)


                # print a summary of the model

                if convolutional:
                    summary(model,[(1,28,28),(1,28,28)])
                else:
                    summary(model, [(784, 2), (784, 1)])


                # define the directories
                model_save_dir = ["saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + ("_frozen" if freeze_weights else "") + ("_cheat_validation/" if cheat_validation else "/")  + str(num_samples) + "S/", model_name, "/",model_name,"_",model_size,"-","","E",".pth"]
                train_loss_dir_txt = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + ("_frozen" if freeze_weights else "") + ("_cheat_validation/" if cheat_validation else "/")  + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + "_train.txt"
                validation_loss_dir_txt = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + ("_frozen" if freeze_weights else "") + ("_cheat_validation/" if cheat_validation else "/")  + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + "_validation.txt"
                loss_dir_plot = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + ("_frozen" if freeze_weights else "") + ("_cheat_validation/" if cheat_validation else "/")  + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ".svg"
                accuracies_dir_txt = "saved_models/MNIST/supervised" + ("_semantics" if semantics else "") + ("_frozen" if freeze_weights else "") + ("_cheat_validation/" if cheat_validation else "/")  + "accuracies/" + model_name + "_" + model_size + ".txt"

                # create directories for the checkpoints and loss files if they don't exist yet
                dir_to_create = "".join(model_save_dir[:3]) + "loss/"
                os.makedirs(dir_to_create,exist_ok=True)

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
                    model.load_state_dict(torch.load(load_dir,map_location=device))
                else:
                    # if train from scratch, check if a loss file already exists
                    assert not(os.path.isfile(train_loss_dir_txt)), "The corresponding loss file already exists, please remove it to train from scratch: " + loss_dir_txt

                if train:
                    avg_loss_per_epoch = train_sup(train_data, model, epochs, model_save_dir, train_loss_dir_txt, validation_data=validation_data, validation_loss_dir_txt=validation_loss_dir_txt, convolutional=convolutional, save_freq=save_freq, epoch_start=epoch_start, device=device, learning_rate=learning_rate)
                    plot_loss([train_loss_dir_txt,validation_loss_dir_txt],loss_dir_plot)

                if save:
                    save_dir = model_save_dir.copy()
                    save_dir[5] = str(epoch_start + epochs)
                    save_dir = "".join(save_dir)
                    torch.save(model.state_dict(),save_dir)

                if evaluate:
                    # if it is the first iteration
                    if i == 0:
                        assert not(os.path.isfile(accuracies_dir_txt)), "The corresponding accuracies file already exists, please remove it to evaluate the models: " + accuracies_dir_txt

                        # create directories for the accuracy if they don't exist yet
                        dir_to_create = os.path.dirname(accuracies_dir_txt)
                        os.makedirs(dir_to_create,exist_ok=True)

                        # initialize the accuracy file with a line showing the size of the training samples
                        txt = "training sample sizes: " + " ".join([str(x) for x in num_training_samples]) + " \n"
                        with open(accuracies_dir_txt,'w') as f:
                            f.write(txt)

                    # compute the accuracy
                    num_context_points = 28 * 28
                    accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=convolutional,num_context_points=num_context_points, save_freq=save_freq, is_CNP=True)
                    print("Number of samples:",num_samples,"Test accuracy: ", accuracy)

                    # write the accuracy to the text file
                    with open(accuracies_dir_txt, 'a+') as f:
                        f.write('%s\n' % accuracy)








