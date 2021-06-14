import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
from torchsummary import summary
from Train.train_CNP_images import train_joint
from CNPs.create_model import  create_model
from CNPs.modify_model_for_classification import modify_model_for_classification
from Utils.data_loader import load_joint_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss

if __name__ == "__main__":

    # pass the arguments
    assert int(sys.argv[1]) == float(sys.argv[1]), "The number of samples should be an integer but was given " + str(float(sys.argv[1]))
    num_samples = int(sys.argv[1])

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "NP_UG" # one of ["CNP", "ConvCNP", "ConvCNPXL", "UnetCNP", "UnetCNP_restrained"]
    model_size = "" # one of ["LR","small","medium","large"]

    semantics = False # use the ConvCNP and CNP pre-trained with blocks of context pixels, i.e. carry more semantics
    validation_split = 0.1
    min_context_points = 2

    if model_name in ["ConvCNP", "ConvCNPXL"]:
        layer_id = -1
        pooling = "average"
    elif model_name in ["UNetCNP", "UNetCNP_restrained"]:
        layer_id = 4
        pooling = "average"
    else:
        layer_id = None
        pooling = None

    variational = False
    if model_name in ["NP_UG"]:
        variational = True
        std_y = 0.1
        num_samples_expectation = 1
        parallel = True

    print(model_name, model_size)

    # for continued supervised training
    train = True
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 300 # which epoch to start from
    else:
        epoch_start = 0

    # training parameters
    num_training_samples = [10,20,40,60,80,100,600,1000,3000]

    # hyper-parameters
    l_sup = 1000 * (60000 * (1-validation_split))/num_samples
    l_unsup = 1
    alpha = 1 * (60000 * (1-validation_split))/num_samples
    alpha_validation = 1

    batch_size = 64
    learning_rate = 1e-4
    epochs = 300
    save_freq = 20

    # load the supervised set
    train_data, validation_data, test_data, img_height, img_width, num_channels = load_joint_data_as_generator(batch_size, num_samples, validation_split = 0.1)

    if not(variational):
        # create the model
        unsupervised_model, convolutional = create_model(model_name,device=device)

        # modify the model to act as a classifier
        model = modify_model_for_classification(unsupervised_model,model_size,convolutional,freeze=False,
                                                img_height=img_height,img_width=img_width,
                                                num_channels=num_channels, layer_id=layer_id, pooling=pooling)
        model.to(device)
    else:
        model, convolutional = create_model(model_name)
        model.prior.loc = model.prior.loc.to(device) 
        model.prior.scale = model.prior.scale.to(device) 

    # print a summary of the model
    if convolutional:
        summary(model,[(1,28,28),(1,28,28)])
    else:
        summary(model, [(784, 2), (784, 1), (784,2)])

    # define the directories
    model_save_dir = ["saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/", model_name, "/",model_name,"_",model_size,"-","","E" + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else ""),".pth"]
    train_joint_loss_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_joint.txt"
    train_unsup_loss_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_unsup.txt"
    train_accuracy_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_accuracy.txt"
    validation_joint_loss_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_joint.txt"
    validation_unsup_loss_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_unsup.txt"
    validation_accuracy_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_accuracy.txt"
    joint_loss_dir_plot = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "joint.svg"
    unsup_loss_dir_plot = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "unsup.svg"
    accuracy_dir_plot = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "acc.svg"
    accuracies_dir_txt = "saved_models/MNIST/joint" + ("_semantics/" if semantics else "/")  + "accuracies/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + ".txt"
    visualisation_dir = ["saved_models/MNIST/joint", ("_semantics/" if semantics else "/") + str(num_samples) + "S/" + model_name, "/visualisation/",model_name,"_","","E_","","C.svg"]

    # create directories for the checkpoints and loss files if they don't exist yet
    dir_to_create = "".join(model_save_dir[:3]) + "loss/"
    os.makedirs(dir_to_create,exist_ok=True)

    if load:
        load_dir = model_save_dir.copy()
        load_dir[-3] = str(epoch_start)
        load_dir = "".join(load_dir)

        if train:
            # check if the loss file is valid
            with open(train_unsup_loss_dir_txt, 'r') as f:
                nbr_unsup_losses = len(f.read().split())
            with open(train_joint_loss_dir_txt, 'r') as f:
                nbr_joint_losses = len(f.read().split())
            with open(train_accuracy_dir_txt, 'r') as f:
                nbr_accuracy = len(f.read().split())

            assert nbr_unsup_losses == epoch_start and nbr_joint_losses == epoch_start and nbr_accuracy == epoch_start, "The number of lines in (one or more of) the joint or unsupervised loss, or the accuracy files does not correspond to the number of epochs"

        # load the model
        model.load_state_dict(torch.load(load_dir,map_location=device))
    else:
        # if train from scratch, check if a loss file already exists
        assert not(os.path.isfile(train_unsup_loss_dir_txt)), "The corresponding unsupervised loss file already exists, please remove it to train from scratch: " + train_unsup_loss_dir_txt
        assert not(os.path.isfile(train_joint_loss_dir_txt)), "The corresponding joint loss file already exists, please remove it to train from scratch: " + train_joint_loss_dir_txt
        assert not (os.path.isfile(train_accuracy_dir_txt)), "The corresponding accuracy file already exists, please remove it to train from scratch: " + train_accuracy_dir_txt
        assert not (os.path.isfile(validation_unsup_loss_dir_txt)), "The corresponding unsupervised loss file already exists, please remove it to train from scratch: " + validation_unsup_loss_dir_txt
        assert not (os.path.isfile(validation_joint_loss_dir_txt)), "The corresponding joint loss file already exists, please remove it to train from scratch: " + validation_joint_loss_dir_txt
        assert not (os.path.isfile(validation_accuracy_dir_txt)), "The corresponding accuracy file already exists, please remove it to train from scratch: " + validation_accuracy_dir_txt

    if train:
        _,_,_,_ = train_joint(train_data, model, epochs, model_save_dir, train_joint_loss_dir_txt, train_unsup_loss_dir_txt, train_accuracy_dir_txt, validation_data, validation_joint_loss_dir_txt, validation_unsup_loss_dir_txt, validation_accuracy_dir_txt, visualisation_dir, semantics=semantics, convolutional=convolutional, variational=variational, min_context_points=min_context_points, save_freq=save_freq, epoch_start=epoch_start, device=device, learning_rate=learning_rate, l_sup=l_sup, l_unsup=l_unsup, alpha=alpha, alpha_validation=alpha_validation, num_samples_expectation=num_samples_expectation, std_y=std_y, parallel=parallel)
        plot_loss([train_unsup_loss_dir_txt,validation_unsup_loss_dir_txt], unsup_loss_dir_plot)
        plot_loss([train_joint_loss_dir_txt, validation_joint_loss_dir_txt], joint_loss_dir_plot)
        plot_loss([train_accuracy_dir_txt, validation_accuracy_dir_txt], accuracy_dir_plot)

    if save:
        save_dir = model_save_dir.copy()
        save_dir[5] = str(epoch_start + epochs)
        save_dir = "".join(save_dir)
        torch.save(model.state_dict(),save_dir)

    if evaluate:
        # create directories for the accuracy if they don't exist yet
        dir_to_create = os.path.dirname(accuracies_dir_txt)
        os.makedirs(dir_to_create, exist_ok=True)

        # compute the accuracy
        num_context_points = 28 * 28
        accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_accuracy_dir_txt,test_data,device,convolutional=convolutional,num_context_points=num_context_points, save_freq=save_freq, is_CNP=True, best="max")
        print("Number of samples:",num_samples,"Test accuracy: ", accuracy)

        # write the accuracy to the text file
        with open(accuracies_dir_txt, 'a+') as f:
            f.write('%s, %s\n' % (num_samples,accuracy))








