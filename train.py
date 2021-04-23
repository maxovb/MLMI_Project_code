import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from Train.train_CNP_images import train_CNP
from CNPs.create_model import  create_model

def load_data(batch_size=64):
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # wrap an iterable over the datasets
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

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
    model_name = "ConvCNP" # one of ["CNP", "ConvCNP", "ConvCNPXL"]

    train = True
    load = False
    save = True
    if load:
        epoch_start = 200 # which epoch to start from
    else:
        epoch_start = 0
    save_freq = 50 # epoch frequency of saving checkpoints

    # parameters
    batch_size = 64
    epochs = 200
    learning_rate = 1e-4

    # create the models
    model, convolutional = create_model(model_name)
    model.to(device)

    # print a summary of the model
    if convolutional:
        summary(model,[(1,28,28),(1,28,28)])
    else:
        summary(model, [(50, 2), (50, 1), (784,2)])

    # load the MNIST data
    train_data, test_data = load_data(batch_size)

    # directories
    model_save_dir = ["saved_models/MNIST/", model_name, "/",model_name,"_","","E",".pth"]
    loss_dir_txt = "saved_models/MNIST/" + model_name + "/loss/" + model_name + ".txt"
    loss_dir_plot = "saved_models/MNIST/" + model_name + "/loss/" + model_name + ".svg"

    if load:
        load_dir = model_save_dir.copy()
        load_dir[5] = str(epoch_start)
        load_dir = "".join(load_dir)

        # check if the loss file is valid
        with open(loss_dir_txt, 'r') as f:
            nbr_losses = len(f.read().split())
        assert nbr_losses == epoch_start, "The number of lines in the loss file does not correspond to the number of epochs"

        # load the model
        model.load_state_dict(torch.load(load_dir,map_location=device))
    else:
        # if train from scratch, check if a loss file already exists
        assert not(os.path.isfile(loss_dir_txt)), "The corresponding loss file already exists, please remove it to train from scratch: " + loss_dir_txt

    if train:
        avg_loss_per_epoch = train_CNP(train_data, model, epochs, model_save_dir, loss_dir_txt, convolutional=convolutional, save_freq=save_freq, epoch_start=epoch_start, device=device, learning_rate=learning_rate)
        plot_loss(loss_dir_txt,loss_dir_plot)

    if save:
        save_dir = model_save_dir.copy()
        save_dir[5] = str(epoch_start + epochs)
        save_dir = "".join(save_dir)
        torch.save(model.state_dict(),save_dir)





