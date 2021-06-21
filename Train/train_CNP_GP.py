import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from Utils.helper_results import qualitative_evaluation_GP

def train_joint(train_data,model,epochs, model_save_dir, train_joint_loss_dir_txt, train_unsup_loss_dir_txt, train_accuracy_dir_txt, validation_data = None, validation_joint_loss_dir_txt = "", validation_unsup_loss_dir_txt = "", validation_accuracy_dir_txt = "", visualisation_dir = None, variational=False, report_freq = 100, learning_rate=1e-3, weight_decay=1e-5, save_freq = 10, n_best_checkpoint = None, epoch_start = 0, device=torch.device('cpu'), alpha=None, alpha_validation=None, num_samples_expectation=None, std_y=None, parallel=False, weight_ratio=False):

    # pre-allocate memory to store the losses
    avg_train_loss_per_epoch = [[],[]] # [joint,unsupervised]
    avg_train_accuracy_per_epoch = []
    avg_validation_loss_per_epoch = [[],[]]
    avg_validation_accuracy_per_epoch = []
    train_loss_to_write = [[],[]]
    train_accuracy_to_write = []
    validation_loss_to_write = [[],[]]
    validation_accuracy_to_write = []

    # define the optimizer
    opt = torch.optim.Adam(model.parameters(),
                           learning_rate,
                           weight_decay=weight_decay)

    # define the sampling threshold to use if max_percentage_context is None
    threshold = 1 / 3

    for i in range(epochs):
            
        print("Epoch:", i + epoch_start + 1)
        model.train()
        train_losses = [[],[]]
        train_num_correct  = []
        train_totals = []
        iterator = tqdm(train_data)
        for batch_idx, (task, target) in enumerate(iterator):

            target = target.to(device)

            x_context = task["x_context"]
            y_context = task["y_context"]
            x_target = task["x"]
            y_target = task["y"]

            num_context_points = x_context.shape[1]
            num_target_points = x_target.shape[1]


            if weight_ratio:
                scale_sup = num_context_points/num_target_points
                scale_unsup = 1-scale_sup
            else:
                scale_sup, scale_unsup = 1, 1

            if not(variational):
                train_joint_loss, train_sup_loss, train_unsup_loss, accuracy, total = model.joint_train_step(x_context, y_context, x_target, target, y_target, opt, alpha=alpha, scale_sup=scale_sup, scale_unsup=scale_unsup)
            else:
                train_joint_loss, train_sup_loss, train_unsup_loss, accuracy, total = model.joint_train_step(x_context,y_context,x_target,target,y_target,opt,alpha,num_samples_expectation=num_samples_expectation, std_y=std_y, parallel=parallel, scale_sup=scale_sup, scale_unsup=scale_unsup)
            train_losses[0].append(train_joint_loss)
            train_losses[1].append(train_unsup_loss)
            train_num_correct.append(accuracy * total)
            train_totals.append(total)

            if batch_idx == 0 or (batch_idx + 1) % report_freq == 0:  # report the loss
                avg_train_joint_loss = np.array(train_losses[0]).mean()
                avg_train_unsup_loss = np.array(train_losses[1]).mean()
                avg_train_accuracy = np.array(train_num_correct).sum()/np.array(train_totals).sum()
                txt = report_multiple_losses(["Joint", "Unsup", "Accuracy"], [round(avg_train_joint_loss,2), round(avg_train_unsup_loss,2), round(avg_train_accuracy,3)], batch_idx)
                iterator.set_description(txt)
                iterator.refresh()  # to show immediately the update

        # compute the average loss over the epoch and store all losses
        epoch_avg_train_joint_loss = np.array(train_losses[0]).mean()
        epoch_avg_train_unsup_loss = np.array(train_losses[1]).mean()
        total = np.array(train_totals).sum()
        epoch_avg_train_accuracy = np.array(train_num_correct).sum() / total
        avg_train_loss_per_epoch[0].append(epoch_avg_train_joint_loss)
        avg_train_loss_per_epoch[1].append(epoch_avg_train_unsup_loss)
        avg_train_accuracy_per_epoch.append(epoch_avg_train_accuracy)
        train_loss_to_write[0].append(epoch_avg_train_joint_loss)
        train_loss_to_write[1].append(epoch_avg_train_unsup_loss)
        train_accuracy_to_write.append(epoch_avg_train_accuracy)


        # Print the average epoch loss
        print("Average training joint loss:", epoch_avg_train_joint_loss, "unsup loss:", epoch_avg_train_unsup_loss, "accuracy:", epoch_avg_train_accuracy, "total", total)

        # calculate the validation loss
        if validation_data:
            validation_losses = [[],[]]
            validation_num_correct= []
            validation_totals = []
            for batch_idx, (data, target) in enumerate(validation_data):

                x_context = task["x_context"]
                y_context = task["y_context"]
                x_target = task["x"]
                y_target = task["y"]

                num_context_points = x_context.shape[1]
                num_target_points = x_target.shape[1]

                target = target.to(device)

                if not(variational):
                    joint_loss, sup_loss, unsup_loss, accuracy, total = model.joint_loss(x_context, y_context, x_target, target, y_target, alpha=alpha_validation)
                    unsup_loss = joint_loss.item()
                else:
                    obj, sup_loss, unsup_loss, accuracy, total = model.joint_loss(x_context,y_context,x_target,target,y_target,alpha_validation,num_samples_expectation,std_y)

                num_correct = accuracy * total
        
                validation_losses[0].append(joint_loss)
                validation_losses[1].append(unsup_loss)
                validation_num_correct.append(num_correct)
                validation_totals.append(total)

            epoch_avg_validation_joint_loss = np.array(validation_losses[0]).mean()
            epoch_avg_validation_unsup_loss = np.array(validation_losses[1]).mean()
            total = np.array(validation_totals).sum()
            epoch_avg_validation_accuracy = np.array(validation_num_correct).sum() / total

            avg_validation_loss_per_epoch[0].append(epoch_avg_validation_joint_loss)
            avg_validation_loss_per_epoch[1].append(epoch_avg_validation_unsup_loss)
            avg_validation_accuracy_per_epoch.append(epoch_avg_validation_accuracy)
            validation_loss_to_write[0].append(epoch_avg_validation_joint_loss)
            validation_loss_to_write[1].append(epoch_avg_validation_unsup_loss)
            validation_accuracy_to_write.append(epoch_avg_validation_accuracy)

            # Print the average epoch loss
            print("Average validation joint loss:", epoch_avg_validation_joint_loss, "unsup loss:",
                  epoch_avg_validation_unsup_loss, "accuracy:", epoch_avg_validation_accuracy, "total", total)

        # save the checkpoint and losses
        if (i + 1) % save_freq == 0:
            # update the file name
            save_dir = model_save_dir.copy()
            save_dir[-3] = str(epoch_start + i + 1)
            save_dir = "".join(save_dir)

            # save the model
            if not (n_best_checkpoint):
                torch.save(model.state_dict(), save_dir)

            # write the average epoch train losses to the txt file
            values = [train_loss_to_write[0],train_loss_to_write[1],train_accuracy_to_write]
            dirs = [train_joint_loss_dir_txt,train_unsup_loss_dir_txt,train_accuracy_dir_txt]
            train_loss_to_write[0], train_loss_to_write[1], train_accuracy_to_write = write_list_to_files_and_reset(values,dirs)

            # write the average epoch validation loss to the txt file if some validation data is supplied
            if validation_data:
                values = [validation_loss_to_write[0], validation_loss_to_write[1], validation_accuracy_to_write]
                dirs = [validation_joint_loss_dir_txt, validation_unsup_loss_dir_txt, validation_accuracy_dir_txt]
                validation_loss_to_write[0], validation_loss_to_write[1], validation_accuracy_to_write = write_list_to_files_and_reset(values, dirs)

            if visualisation_dir:
                # get the output directory
                for num_context_points in [3,10,50]:
                    visualisation_dir = visualisation_dir.copy()
                    visualisation_dir[5] = str(epoch_start + i + 1)
                    visualisation_dir[-2] = str(num_context_points)

                    # train data
                    visualisation_dir_train = visualisation_dir.copy()
                    visualisation_dir_train[2] += "train/"
                    img_output_dir_train = "".join(visualisation_dir_train)

                    # create directories if it doesn't exist yet
                    dir_to_create = "".join(visualisation_dir_train[:3])
                    os.makedirs(dir_to_create, exist_ok=True)

                    qualitative_evaluation_GP(model, train_data, num_context_points, device=device,
                                              save_dir=img_output_dir_train,variational=variational,
                                              include_class_predictions=True)


                    if validation_data:
                        # validation data
                        visualisation_dir_validation = visualisation_dir.copy()
                        visualisation_dir_validation[2] += "validation/"
                        img_output_dir_validation = "".join(visualisation_dir_validation)

                        # create directories if it doesn't exist yet
                        dir_to_create = "".join(visualisation_dir_validation[:3])
                        os.makedirs(dir_to_create, exist_ok=True)

                        qualitative_evaluation_GP(model, validation_data, num_context_points, device,
                                                  img_output_dir_validation, variational,
                                                  include_class_predictions=True)

        if n_best_checkpoint:
            pass
            # TODO: n-best checkpoint saving

    # write the average epoch train losses to the txt file
    values = [train_loss_to_write[0], train_loss_to_write[1], train_accuracy_to_write]
    dirs = [train_joint_loss_dir_txt, train_unsup_loss_dir_txt, train_accuracy_dir_txt]
    train_loss_to_write[0], train_loss_to_write[1], train_accuracy_to_write = write_list_to_files_and_reset(
        values, dirs)

    # write the average epoch validation loss to the txt file if some validation data is supplied
    if validation_data:
        values = [validation_loss_to_write[0], validation_loss_to_write[1], validation_accuracy_to_write]
        dirs = [validation_joint_loss_dir_txt, validation_unsup_loss_dir_txt, validation_accuracy_dir_txt]
        validation_loss_to_write[0], validation_loss_to_write[1], validation_accuracy_to_write = write_list_to_files_and_reset(values, dirs)

    return avg_train_loss_per_epoch, avg_train_accuracy_per_epoch, avg_validation_loss_per_epoch, avg_validation_accuracy_per_epoch

def report_loss(name,avg_loss,step):
    txt = name + " loss, step " + str(step) + ": " + str(avg_loss)
    return txt

def write_list_to_files_and_reset(list_values,list_dir):
    out = []
    for i,values in enumerate(list_values):
        with open(list_dir[i], 'a+') as f:
            for loss_item in values:
                f.write('%s\n' % loss_item)
        out.append([])
    return out

def report_multiple_losses(names,avg_losses,step):
    txt = "Step: " + str(step)
    for i,name in enumerate(names):
        txt += " " + name + ": " + str(avg_losses[i])
    return txt


