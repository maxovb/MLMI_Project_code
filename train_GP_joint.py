import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
import stheno
from torchsummary import summary
from Train.train_CNP_GP import train_joint
from CNPs.create_model_1D import  create_model_off_the_grid
from CNPs.modify_model_for_classification import modify_model_for_classification_off_the_grid
from data.GP.GP_data_generator import MultiClassGPGenerator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss


if __name__ == "__main__":

    data_name = "GP"

    list_kernels = [stheno.EQ().stretch(1),stheno.EQ().stretch(1/2),stheno.EQ().periodic(1),stheno.EQ().periodic(3/2)]
    kernel_names = ["EQ 1", "EQ 1/2", "Periodic 1", "Periodic 3/2"]
    num_classes = len(list_kernels)

    # pass the arguments
    assert float(sys.argv[1]) > 0 and float(sys.argv[1]) <= 1, "The number of samples should be a percentage but was given " + str(float(sys.argv[1]))
    percentage_label = float(sys.argv[1])

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    model_name = "UNetCNP_restrained_GMM" # one of ["CNP", "ConvCNP", "ConvCNPXL", "UnetCNP", "UnetCNP_restrained", "UNetCNP_GMM","UNetCNP_restrained_GMM"]
    model_size = "LR" # one of ["LR","small","medium","large"]

    weight_ratio = True # weight the loss with the ratio of context pixels present

    train = True
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 20 # which epoch to start from
    else:
        epoch_start = 0

    batch_size = 64
    num_tasks = 850
    num_batches_per_epoch = 256
    learning_rate = 1e-4
    epochs = 200 - epoch_start
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
    if model_name in ["UNetCNP_GMM", "UNetCNP_restrained_GMM", "UNetCNP_GMM_blocked", "UNetCNP_restrained_GMM_blocked"]:
        mixture = True
        model_size_creation = model_size

    print(model_name, model_size)

    # hyper-parameters
    if not (variational):
        if not (mixture):
            alpha = 1000 * 1/percentage_label
            alpha_validation = 1000
        else:
            alpha = 1/percentage_label
            alpha_validation = 1
    else:
        alpha = 1/percentage_label
        alpha_validation = 1

    # load the supervised set
    train_data = MultiClassGPGenerator(list_kernels,percentage_label, kernel_names=kernel_names, batch_size=batch_size, num_tasks=num_tasks)
    test_data = MultiClassGPGenerator(list_kernels, 1, kernel_names=kernel_names, batch_size=batch_size, num_tasks=num_tasks)

    if not(variational):
        if not(mixture):
            # create the model
            unsupervised_model = create_model_off_the_grid(model_name,num_classes=num_classes)

            # modify the model to act as a classifier
            model = modify_model_for_classification_off_the_grid(unsupervised_model,model_size,freeze=False,
                                                                 layer_id=layer_id, pooling=pooling)
            model.to(device)
        else:
            model = create_model_off_the_grid(model_name, model_size_creation,num_classes=num_classes)
            model.to(device)
    else:
        model = create_model_off_the_grid(model_name,num_classes=num_classes)
        model.to(device)
        model.prior.loc = model.prior.loc.to(device)
        model.prior.scale = model.prior.scale.to(device)

    # define the directories
    experiment_dir_list = ["saved_models/" + data_name +  "/joint/" + str(percentage_label) + "P/", model_name, "/"]
    experiment_dir_txt = "".join(experiment_dir_list)
    model_save_dir = experiment_dir_list + [model_name, "_", model_size, "-", "", "E" + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else ""), ".pth"]
    train_joint_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_joint.txt"
    train_unsup_loss_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_unsup.txt"
    train_accuracy_dir_txt = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "_train_accuracy.txt"
    joint_loss_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "joint.svg"
    unsup_loss_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "unsup.svg"
    accuracy_dir_plot = experiment_dir_txt + "loss/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + "acc.svg"
    visualisation_dir = experiment_dir_list[:-1] + ["/visualisation/", model_name, "_", "", "E_", "", "C.svg"]
    accuracies_dir_txt = "saved_models/" + data_name +  "/joint" + "accuracies/" + model_name + "_" + model_size + ("_" + str(layer_id) + "L_" + pooling if layer_id and pooling else "") + ".txt"

    # create directories for the checkpoints and loss files if they don't exist yet
    dir_to_create = "".join(model_save_dir[:3]) + "loss/"
    os.makedirs(dir_to_create, exist_ok=True)

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
        model.load_state_dict(torch.load(load_dir, map_location=device))

    else:
        # if train from scratch, check if a loss file already exists
        assert not (os.path.isfile(train_unsup_loss_dir_txt)), "The corresponding unsupervised loss file already exists, please remove it to train from scratch: " + train_unsup_loss_dir_txt
        assert not (os.path.isfile(train_joint_loss_dir_txt)), "The corresponding joint loss file already exists, please remove it to train from scratch: " + train_joint_loss_dir_txt
        assert not (os.path.isfile(train_accuracy_dir_txt)), "The corresponding accuracy file already exists, please remove it to train from scratch: " + train_accuracy_dir_txt

    if train:
        _, _, _, _ = train_joint(train_data, model, epochs, model_save_dir, train_joint_loss_dir_txt,
                                 train_unsup_loss_dir_txt, train_accuracy_dir_txt,
                                 visualisation_dir=visualisation_dir, variational=variational,
                                 save_freq=save_freq, epoch_start=epoch_start, device=device,
                                 learning_rate=learning_rate, alpha=alpha, alpha_validation=alpha_validation,
                                 num_samples_expectation=num_samples_expectation, std_y=std_y, parallel=parallel,
                                 weight_ratio=weight_ratio)
        plot_loss([train_unsup_loss_dir_txt], unsup_loss_dir_plot)
        plot_loss([train_joint_loss_dir_txt], joint_loss_dir_plot)
        plot_loss([train_accuracy_dir_txt], accuracy_dir_plot)

    if save:
        save_dir = model_save_dir.copy()
        save_dir[5] = str(epoch_start + epochs)
        save_dir = "".join(save_dir)
        torch.save(model.state_dict(), save_dir)

    if evaluate:
        # create directories for the accuracy if they don't exist yet
        dir_to_create = os.path.dirname(accuracies_dir_txt)
        os.makedirs(dir_to_create, exist_ok=True)


        accuracy = test_model_accuracy_with_best_checkpoint(model, model_save_dir, train_accuracy_dir_txt,
                                                            test_data, device, convolutional=convolutional,
                                                            num_context_points=num_context_points,
                                                            save_freq=save_freq, is_CNP=True, best="max")
        print("Number of samples:", num_samples, "Test accuracy: ", accuracy)

        # write the accuracy to the text file
        with open(accuracies_dir_txt, 'a+') as f:
            f.write('%s, %s\n' % (num_samples, accuracy))



