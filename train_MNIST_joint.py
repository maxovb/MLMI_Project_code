import numpy as np
import random
import matplotlib.pyplot as plt
import os
import torch
import sys
import math
from torchsummary import summary
from Train.train_CNP_images import train_joint
from CNPs.create_model import  create_model
from CNPs.modify_model_for_classification import modify_model_for_classification
from Utils.data_loader import load_joint_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss
from Utils.helpers_train import GradNorm

if __name__ == "__main__":

    random.seed(1234)

    # pass the arguments
    assert int(sys.argv[1]) == float(sys.argv[1]), "The number of samples should be an integer but was given " + str(float(sys.argv[1]))
    num_samples = int(sys.argv[1])

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "UNetCNP" # one of ["CNP", "ConvCNP", "ConvCNPXL", "UnetCNP", "UnetCNP_restrained", "UNetCNP_GMM","UNetCNP_restrained_GMM"]
    model_size = "medium_dropout" # one of ["LR","small","medium","large"]
    block_connections = False  # whether to block the skip connections at the middle layers of the UNet

    semantics = True # use the ConvCNP and CNP pre-trained with blocks of context pixels, i.e. carry more semantics
    weight_ratio = True # weight the loss with the ratio of context pixels present in the image
    consistency_regularization = True# whether to use consistency regularization or not
    grad_norm = True # whether to use GradNorm to balance the losses
    classify_same_image = True # whether to augment the tarinign with an extra task where the model discriminates between two disjoint set of context pixels as coming from the same image or not
    validation_split = 0.1
    min_context_points = 2


    # for continued supervised training
    train = True
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 100 # which epoch to start from
    else:
        epoch_start = 0

    batch_size = 64
    learning_rate = 2e-4
    epochs = 400 - epoch_start
    save_freq = 20

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
    else:
        std_y = None
        num_samples_expectation = None
        parallel = None

    mixture = False
    model_size_creation = None
    if model_name in ["UNetCNP_GMM","UNetCNP_restrained_GMM","UNetCNP_GMM_blocked","UNetCNP_restrained_GMM_blocked"]:
        mixture = True
        model_size_creation = model_size

    print(model_name, model_size)

    # training parameters
    num_training_samples = [10,20,40,60,80,100,600,1000,3000]

    # hyper-parameters
    if not(variational):
        if mixture:
            alpha = 789 * (60000 * (1-validation_split))/num_samples
            alpha_validation = 789
        else:
            if grad_norm:
                alpha = 1
                alpha_validation = 1
            else:
                alpha = (60000 * (1-validation_split))/num_samples
                alpha_validation = 1
    else:
        alpha = 1 * (60000 * (1-validation_split))/num_samples
        alpha_validation = 1
    
    # load the supervised set
    train_data, validation_data, test_data, img_height, img_width, num_channels = load_joint_data_as_generator(batch_size, num_samples, validation_split = 0.1)

    if not(variational):
        if not(mixture):
            # create the model
            unsupervised_model, convolutional = create_model(model_name)

            # modify the model to act as a classifier
            model = modify_model_for_classification(unsupervised_model,model_size,convolutional,freeze=False,
                                                    img_height=img_height,img_width=img_width,
                                                    num_channels=num_channels, layer_id=layer_id, pooling=pooling,
                                                    classify_same_image=classify_same_image)
            model.to(device)
        else:
            model, convolutional = create_model(model_name, model_size_creation, classify_same_image=classify_same_image)
            model.to(device)
    else:
        model, convolutional = create_model(model_name, classify_same_image=classify_same_image)
        model.prior.loc = model.prior.loc.to(device) 
        model.prior.scale = model.prior.scale.to(device) 

    # print a summary of the model
    """
    if convolutional:
        summary(model,[(1,28,28),(1,28,28)])
    else:
        summary(model, [(784, 2), (784, 1), (784,2)])
    """
    num_losses = 2
    theoretical_minimum_loss = [- img_width * img_height * num_channels * math.log(1/(math.sqrt(2*math.pi)*0.01))] # reconstruction loss
    if consistency_regularization:
        num_losses += 1
        theoretical_minimum_loss.append(-math.log(2))
    if classify_same_image:
        num_losses += 1
        theoretical_minimum_loss.append(0)
    theoretical_minimum_loss.append(0) # cross-entropy

    ratios = [1 for _ in range(num_losses)]
    ratios[-1] = (60000 * (1 - validation_split)) / num_samples

    if grad_norm:
        gamma = 1.5 # hyper-parameter for grad_norm
        grad_norm_iterator = GradNorm(model,gamma,ratios,theoretical_minimum_loss)
    else:
        grad_norm_iterator = None

    # define the directories
    experiment_dir_list = ["saved_models/MNIST/joint" + ("_semantics" if semantics else "_") + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") + ("_ET/" if classify_same_image else "/") + str(num_samples) + "S/", model_name, "/"]
    experiment_dir_txt = "".join(experiment_dir_list)
    model_save_dir = experiment_dir_list + [model_name,"_",model_size,"-","","E" + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else ""),".pth"]
    train_joint_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_joint.txt"
    train_unsup_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_unsup.txt"
    train_accuracy_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_accuracy.txt"
    train_accuracy_discriminator_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_accuracy_discriminator.txt"
    validation_joint_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_joint.txt"
    validation_unsup_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_unsup.txt"
    validation_accuracy_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_accuracy.txt"
    validation_accuracy_discriminator_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_validation_accuracy_discriminator.txt"
    joint_loss_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "joint.svg"
    unsup_loss_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "unsup.svg"
    accuracy_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "acc.svg"
    visualisation_dir = experiment_dir_list[:-1] + ["/visualisation/",model_name,"_","","E_","","C.svg"]
    gradnorm_dir_txt = experiment_dir_txt + "grad_norm/"
    accuracies_dir_txt = "saved_models/MNIST/joint" + ("_semantics" if semantics else "") + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") + ("_ET/" if classify_same_image else "/") + "accuracies/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + ".txt"

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
        _,_,_,_ = train_joint(train_data, model, epochs, model_save_dir, train_joint_loss_dir_txt,
                              train_unsup_loss_dir_txt, train_accuracy_dir_txt, validation_data,
                              validation_joint_loss_dir_txt, validation_unsup_loss_dir_txt, validation_accuracy_dir_txt,
                              visualisation_dir, semantics=semantics, convolutional=convolutional,
                              variational=variational, min_context_points=min_context_points, save_freq=save_freq,
                              epoch_start=epoch_start, device=device, learning_rate=learning_rate, alpha=alpha,
                              alpha_validation=alpha_validation, num_samples_expectation=num_samples_expectation,
                              std_y=std_y, parallel=parallel, weight_ratio=weight_ratio,
                              consistency_regularization=consistency_regularization,
                              grad_norm_iterator=grad_norm_iterator, gradnorm_dir_txt=gradnorm_dir_txt,
                              classify_same_image=classify_same_image,
                              train_accuracy_discriminator_dir_txt=train_accuracy_discriminator_dir_txt,
                              validation_accuracy_discriminator_dir_txt=validation_accuracy_discriminator_dir_txt)
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








