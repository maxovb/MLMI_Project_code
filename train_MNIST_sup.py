import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
from Train.train_CNP_images import train_CNP_sup
from CNPs.create_model import  create_model
from CNPs.modify_model_for_classification import modify_model_for_classification
from Utils.data_processor import image_processor

def load_supervised_data(batch_size=64,num_training_samples=100):
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

    num_classes = max(training_data.targets).item() + 1
    assert num_training_samples % num_classes == 0, "The number of training samples ("  + str(num_training_samples) \
                                                  +") must be divisible by the number of classes (" + str(num_classes)\
                                                  +")"

    num_samples_per_class = num_training_samples//num_classes

    num_validation_samples = num_training_samples//10
    num_validation_samples_per_class = max(num_validation_samples//num_classes,1)

    # separate the data per class
    data_divided_per_class = [[] for _ in range(num_classes)]
    for (img,label) in training_data:
        data_divided_per_class[label].append((img,label))

    # shuffle all lists and select a subset
    selected_training_data = []
    selected_validation_data = []
    for j in range(num_classes):
        random.shuffle(data_divided_per_class[j])
        selected_training_data.extend(data_divided_per_class[j][:num_samples_per_class])
        selected_validation_data.extend(data_divided_per_class[j][num_samples_per_class:num_samples_per_class+num_validation_samples_per_class])

    # shuffle the new dataset once last time to avoid the classes between clustered together
    random.shuffle(selected_training_data)
    random.shuffle(selected_validation_data)

    # wrap an iterable over the datasets
    train_dataloader = DataLoader(selected_training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(selected_validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    # get img_width and height
    img_height, img_width = train_dataloader.dataset[0][0].shape[1], train_dataloader.dataset[0][0].shape[2]
    return train_dataloader, validation_dataloader, test_dataloader, img_height, img_width

def test_model_accuracy(model,test_data,convolutional,num_context_points):
    sum, total = 0,0
    for i,(data,target) in enumerate(test_data):
        if convolutional:
            mask, context_img = image_processor(data, num_context_points, convolutional, device)
            data = data.to(device)
            batch_accuracy, batch_size = model.evaluate_accuracy(x_context,y_context,target)
        else:
            x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional,
                                                                       device)
            batch_accuracy, batch_size = model.evaluate_accuracy(x_context,y_context,target)
        sum += batch_accuracy * batch_size
        total += batch_size
    return sum/total


def plot_loss(list_loss_dir_txt,loss_dir_plot):
    l = len(list_loss_dir_txt)
    losses = [[] for _ in range(l)]
    for i,filename in enumerate(list_loss_dir_txt):
        with open(filename,"r") as f:
            for x in f.read().split():
                if x != "":
                    losses[i].append(float(x))

    # plot
    plt.figure()
    for i in range(l):
        plt.plot(np.arange(1,len(losses[i])+1),losses[i])
    if l == 2:
        plt.legend(["Train loss", "Validation loss"])
    plt.xlabel("Epoch",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.savefig(loss_dir_plot)

def load_unsupervised_model(model_name, epoch,device):
    model_load_dir = ["saved_models/MNIST/", model_name, "/", model_name, "_", str(epoch), "E", ".pth"]
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
    model_name = "ConvCNP" # one of ["CNP", "ConvCNP", "ConvCNPXL"]

    # for continued supervised training
    train = False
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 100 # which epoch to start from
    else:
        epoch_start = 0
    save_freq = 20 # epoch frequency of saving checkpoints

    # parameters from the model to load
    epoch_unsup = 100 # unsupervised model to load intially
    freeze_weights = True # freeze the weights of the part taken from the unsupervised model

    # training parameters
    num_training_samples = [100,600,1000,3000]
    batch_size = 8
    epochs = 200
    learning_rate = 1e-3

    for num_samples in num_training_samples:
        # load the supervised set
        train_data, validation_data, test_data, img_height, img_width = load_supervised_data(batch_size, num_samples)


        # create the model
        model, convolutional = load_unsupervised_model(model_name,epoch_unsup,device)

        # modify the model to act as a classifier
        model = modify_model_for_classification(model,convolutional,freeze_weights,img_height=img_height,img_width=img_width)
        model.to(device)

        # print a summary of the model
        if convolutional:
            summary(model,[(1,28,28),(1,28,28)])
        else:
            summary(model, [(784, 2), (784, 1)])

        # define the directories
        model_save_dir = ["saved_models/MNIST/supervised/" + str(num_samples) + "S/", model_name, "/",model_name,"_","","E",".pth"]
        train_loss_dir_txt = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "train.txt"
        validation_loss_dir_txt = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "validation.txt"
        loss_dir_plot = "saved_models/MNIST/supervised/" + str(num_samples) + "S/" + model_name + "/loss/" + model_name + ".svg"

        if load:
            load_dir = model_save_dir.copy()
            load_dir[5] = str(epoch_start)
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
            avg_loss_per_epoch = train_CNP_sup(train_data, model, epochs, model_save_dir, train_loss_dir_txt, validation_data=validation_data, validation_loss_dir_txt=validation_loss_dir_txt, convolutional=convolutional, save_freq=save_freq, epoch_start=epoch_start, device=device, learning_rate=learning_rate)
            plot_loss([train_loss_dir_txt,validation_loss_dir_txt],loss_dir_plot)

        if save:
            save_dir = model_save_dir.copy()
            save_dir[5] = str(epoch_start + epochs)
            save_dir = "".join(save_dir)
            torch.save(model.state_dict(),save_dir)

        if evaluate:
            num_context_points = 28 * 28
            accuracy = test_model_accuracy(model,test_data,convolutional,num_context_points)
            print("Number of samples:",num_samples,"Test accuracy: ", accuracy)






