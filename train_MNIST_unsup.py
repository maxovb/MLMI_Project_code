import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from Train.train_CNP_images import train_CNP_unsup
from CNPs.create_model import  create_model
from Utils.data_loader import load_data_unsupervised
from Utils.helper_results import qualitative_evaluation_images

def plot_loss(loss_dir_txt,loss_dir_plot):
        loss = []
        with open(loss_dir_txt,"r") as f:
            for x in f.read().split():
                if x != "":
                    loss.append(int(float(x)))

        # plot
        plt.figure()
        plt.plot(np.arange(1,len(loss)+1),loss)
        plt.xlabel("Epoch",fontsize=15)
        plt.ylabel("Negative log-likelihood",fontsize=15)
        plt.savefig(loss_dir_plot)


if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "CNP" # one of ["CNP", "ConvCNP", "ConvCNPXL"]

    train = True
    load = False
    save = False
    if load:
        epoch_start = 200 # which epoch to start from
    else:
        epoch_start = 0
    save_freq = 50 # epoch frequency of saving checkpoints

    # parameters
    batch_size = 64
    validation_split = 0.10
    learning_rate = 1e-4
    if train:
        epochs = 400
    else:
        epochs = 0

    # create the models
    model, convolutional = create_model(model_name)
    model.to(device)

    # print a summary of the model
    if convolutional:
        summary(model,[(1,28,28),(1,28,28)])
    else:
        summary(model, [(50, 2), (50, 1), (784,2)])

    # load the MNIST data
    train_data, valid_data, test_data = load_data_unsupervised(batch_size,validation_split=validation_split)

    # directories
    model_save_dir = ["saved_models/MNIST/", model_name, "/",model_name,"_","","E",".pth"]
    train_loss_dir_txt = "saved_models/MNIST/" + model_name + "/loss/train_" + model_name + ".txt"
    validation_loss_dir_txt = "saved_models/MNIST/" + model_name + "/loss/validation_" + model_name + ".txt"
    loss_dir_plot = "saved_models/MNIST/" + model_name + "/loss/" + model_name + ".svg"
    visualisation_dir = ["saved_models/MNIST/", model_name, "/visualisation/",model_name,"_","","E_","","C.svg"]

    if load:
        load_dir = model_save_dir.copy()
        load_dir[5] = str(epoch_start)
        load_dir = "".join(load_dir)

        # check if the loss file is valid
        with open(train_loss_dir_txt, 'r') as f:
            nbr_losses = len(f.read().split())
        assert nbr_losses == epoch_start, "The number of lines in the loss file does not correspond to the number of epochs"

        # load the model
        model.load_state_dict(torch.load(load_dir,map_location=device))
    else:
        # if train from scratch, check if a loss file already exists
        if train or save:
            assert not(os.path.isfile(train_loss_dir_txt)), "The corresponding loss file already exists, please remove it to train from scratch: " + train_loss_dir_txt

    if train:
        avg_loss_per_epoch = train_CNP_unsup(train_data, model, epochs, model_save_dir, train_loss_dir_txt, validation_data=valid_data, validation_loss_dir_txt=validation_loss_dir_txt, convolutional=convolutional, visualisation_dir=visualisation_dir, save_freq=save_freq, epoch_start=epoch_start, device=device, learning_rate=learning_rate)
        plot_loss(train_loss_dir_txt,loss_dir_plot)

    if save:
        save_dir = model_save_dir.copy()
        save_dir[5] = str(epoch_start + epochs)
        save_dir = "".join(save_dir)
        torch.save(model.state_dict(),save_dir)







