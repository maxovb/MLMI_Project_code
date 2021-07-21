import numpy as np
import random
import matplotlib.pyplot as plt
import os
import torch
import sys
import math
import time
import argparse
from torchsummary import summary
from Train.train_CNP_images import train_joint
from CNPs.create_model import  create_model
from CNPs.modify_model_for_classification import modify_model_for_classification
from Utils.data_loader import load_joint_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, LossWriter, plot_losses_from_loss_writer, InfoWriter, evaluate_model_full_accuracy
from Utils.helpers_train import GradNorm

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("n", help="Number of labelled samples", type=int)

    # Optional arguments
    parser.add_argument("-CL", "--consitencyloss", help="Use consistency loss", type=str, default="False")
    parser.add_argument("-ET", "--extratask", help="Use extra classification task", type=str, default="False")
    parser.add_argument("-GN", "--gradnorm", help="Use grad norm", type=str, default="False")
    parser.add_argument("-R","--ratiodivide",help="Value by which to divide the ratio for the grad loss",type=float,default=1)

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    random.seed(1234)

    # pass the arguments
    args = parseArguments()

    num_samples = args.n
    assert int(num_samples) == float(num_samples), "The number of samples should be an integer but was given " + str(float(sys.argv[1]))

    consistency_regularization = args.consitencyloss.lower() == "true"
    classify_same_image = args.extratask.lower() == "true"
    grad_norm = args.gradnorm.lower() == "true"
    R = args.ratiodivide

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # percentage of unlabelled imagesc
    percentage_unlabelled_set = 1
    data_version = 0

    # type of model
    model_name = "UNetCNP" # one of ["CNP", "ConvCNP", "ConvCNPXL", "UnetCNP", "UnetCNP_restrained", "UNetCNP_GMM","UNetCNP_restrained_GMM"]
    model_size = "medium_dropout" # one of ["LR","small","medium","large"]
    block_connections = False  # whether to block the skip connections at the middle layers of the UNet

    semantics = True # use the ConvCNP and CNP pre-trained with blocks of context pixels, i.e. carry more semantics
    weight_ratio = True # weight the loss with the ratio of context pixels present in the image
    #consistency_regularization = False # whether to use consistency regularization or not
    #grad_norm = False # whether to use GradNorm to balance the losses
    #classify_same_image = False # whether to augment the tarinign with an extra task where the model discriminates between two disjoint set of context pixels as coming from the same image or not
    validation_split = 0.1
    min_context_points = 2

    # for continued supervised training
    train = False
    load = True
    save = False
    evaluate = True
    if load:
        epoch_start = 2000 # which epoch to start from
    else:
        epoch_start = 0

    if percentage_unlabelled_set < 0.25:
        batch_size = 16
    else:
        batch_size = 64
    learning_rate = 2e-4
    epochs = 2000 - epoch_start
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
    print("CL",consistency_regularization,"GN",grad_norm,"ET",classify_same_image)
    print("n",num_samples)

    # training parameters
    num_training_samples = [10,20,40,60,80,100,600,1000,3000]

    # hyper-parameters
    if not(variational):
        if mixture:
            if grad_norm:
                alpha = 1
                alpha_validation = 1
            else:
                alpha = (60000 * percentage_unlabelled_set * (1-validation_split))/num_samples / R
                alpha_validation = 1
        else:
            if grad_norm:
                alpha = 1
                alpha_validation = 1
            else:
                alpha = (60000 * percentage_unlabelled_set * (1-validation_split))/num_samples / R
                alpha_validation = 1
    else:
        alpha = 1 * (60000 * percentage_unlabelled_set * (1-validation_split))/num_samples / R
        alpha_validation = 1
    
    # load the supervised set
    out = load_joint_data_as_generator(batch_size,num_samples,
                                       validation_split = 0.1,
                                       percentage_unlabelled_set = percentage_unlabelled_set,
                                       data_version = data_version)
    train_data, validation_data, test_data, img_height, img_width, num_channels = out

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
    losses_name = ["Regression"]
    acronym_losses = ["rec_loss"]
    initial_loss = [0]
    if consistency_regularization:
        num_losses += 1
        theoretical_minimum_loss.append(-math.log(2))
        losses_name.append("Consistency")
        acronym_losses.append("cons_loss")
        initial_loss.append(0)
    if classify_same_image:
        num_losses += 1
        theoretical_minimum_loss.append(0)
        losses_name.append("Extra classification task")
        acronym_losses.append("loss_discriminator")
        initial_loss.append(0.3)

    theoretical_minimum_loss.append(0) # cross-entropy
    losses_name.append("Supervised")
    acronym_losses.append("sup_loss")
    initial_loss.append(1.3)

    ratios = [1 for _ in range(num_losses)]
    ratios[-1] = (60000 * percentage_unlabelled_set * (1 - validation_split)) / num_samples / R

    if grad_norm:
        if load:
            initial_task_loss = np.array(initial_loss)
        else:
            initial_task_loss = None
        gamma = 1.5 # hyper-parameter for grad_norm
        grad_norm_iterator = GradNorm(model,gamma,ratios,theoretical_minimum_loss,losses_name=losses_name,initial_task_loss=initial_task_loss)
    else:
        grad_norm_iterator = None

    # define the directories
    experiment_dir_list = ["saved_models/MNIST/joint_" + str(R) + "R" + ("_semantics" if semantics else "_") + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") + ("_ET/" if classify_same_image else "/") + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/" + str(num_samples) + "S/", model_name, "/"]
    experiment_dir_txt = "".join(experiment_dir_list)
    info_dir_txt = experiment_dir_txt + "information.txt"
    model_save_dir = experiment_dir_list + [model_name,"_",model_size,"-","","E" + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else ""),".pth"]
    visualisation_dir = experiment_dir_list[:-1] + ["/visualisation/",model_name,"_","","E_","","C.svg"]
    gradnorm_dir_txt = experiment_dir_txt + "grad_norm/"
    accuracies_dir_txt = "saved_models/MNIST/joint_" + str(R) + "R" +("_semantics" if semantics else "") + ("_cons" if consistency_regularization else "") + ("_GN_" + str(gamma) + "" if grad_norm else "") + ("_ET/" if classify_same_image else "/") + "accuracies/" + str(percentage_unlabelled_set) + "P_" + str(data_version) + "V/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + ".txt"


    train_losses_dir_list = [experiment_dir_txt + "loss/" + model_name + "_" + model_size +
                             ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "_")
                             + "_train_","",".txt"]
    validation_losses_dir_list = [experiment_dir_txt + "loss/" + model_name + "_" + model_size +
                                  ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "_")
                                  + "_validation_", "", ".txt"]

    # for evaluating the true model accuracy at the saved checkpoints
    loss_train_full_accuracies_dir_list = train_losses_dir_list.copy()
    loss_train_full_accuracies_dir_list[1] = "full_accuracy"
    loss_train_full_accuracies_dir_txt = "".join(loss_train_full_accuracies_dir_list)
    loss_validation_full_accuracies_dir_list = validation_losses_dir_list.copy()
    loss_validation_full_accuracies_dir_list[1] = "full_accuracy"
    loss_validation_full_accuracies_dir_txt = "".join(loss_validation_full_accuracies_dir_list)

    # create the loss_writers
    train_loss_writer = LossWriter(train_losses_dir_list)
    validation_loss_writer = LossWriter(validation_losses_dir_list)

    # create the info_writer
    info_writer = InfoWriter(info_dir_txt)

    # get the number of parameters
    n = int(model.num_params)
    info_writer.update_number_of_parameters(n)

    # create directories for the checkpoints and loss files if they don't exist yet
    dir_to_create = "".join(model_save_dir[:3]) + "loss/"
    os.makedirs(dir_to_create,exist_ok=True)

    if load:
        load_dir = model_save_dir.copy()
        load_dir[-3] = str(epoch_start)
        load_dir = "".join(load_dir)

        if train:
            # check if the loss file is valid
            with open(train_loss_writer.obtain_loss_dir_txt("joint_loss"), 'r') as f:
                nbr_joint_losses = len(f.read().split())

            assert nbr_joint_losses == epoch_start, "The number of lines in (one or more of) the joint or unsupervised loss, or the accuracy files does not correspond to the number of epochs"

        # load the model
        model.load_state_dict(torch.load(load_dir,map_location=device))
    else:
        # if train from scratch, check if a loss file already exists
        assert not(os.path.isfile(train_loss_writer.obtain_loss_dir_txt("joint_loss"))), "The corresponding unsupervised loss file already exists, please remove it to train from scratch: " + train_loss_writer.obtain_loss_dir_txt("joint_loss")

    if train:
        t0 = time.time()
        train_joint(train_data, model, epochs, model_save_dir, train_loss_writer, validation_data,
                    validation_loss_writer, visualisation_dir, semantics=semantics, convolutional=convolutional,
                    variational=variational, min_context_points=min_context_points, save_freq=save_freq,
                    epoch_start=epoch_start, device=device, learning_rate=learning_rate, alpha=alpha,
                    alpha_validation=alpha_validation, num_samples_expectation=num_samples_expectation,
                    std_y=std_y, parallel=parallel, weight_ratio=weight_ratio,
                    consistency_regularization=consistency_regularization,
                    grad_norm_iterator=grad_norm_iterator, gradnorm_dir_txt=gradnorm_dir_txt,
                    classify_same_image=classify_same_image)
        t = time.time() - t0
        info_writer.update_time(t)
        plot_losses_from_loss_writer(train_loss_writer, validation_loss_writer)
        evaluate_model_full_accuracy(model, experiment_dir_txt, loss_train_full_accuracies_dir_txt, train_data, device,
                                     convolutional=convolutional)
        evaluate_model_full_accuracy(model, experiment_dir_txt, loss_validation_full_accuracies_dir_txt, validation_data,
                                     device, convolutional=convolutional)

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
        accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_writer.obtain_loss_dir_txt("accuracy"),test_data,device,convolutional=convolutional,num_context_points=num_context_points, save_freq=save_freq, is_CNP=True, best="max")
        print("Number of samples:",num_samples,"Test accuracy: ", accuracy)

        # write the accuracy to the text file
        with open(accuracies_dir_txt, 'a+') as f:
            f.write('%s, %s\n' % (num_samples,accuracy))



