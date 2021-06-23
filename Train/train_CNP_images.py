import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from Utils.data_processor import image_processor, format_context_points_image
from Utils.helper_results import qualitative_evaluation_images

def train_CNP_unsup(train_data,model,epochs, model_save_dir, train_loss_dir_txt, semantics=False, validation_data=None, validation_loss_dir_txt="",min_context_points=2,max_percentage_context = None, convolutional=False, visualisation_dir=None, report_freq = 100, learning_rate=1e-3, weight_decay=1e-5, save_freq = 10, epoch_start = 0, device=torch.device('cpu')):
    img_height, img_width = train_data.dataset[0][0].shape[1], train_data.dataset[0][0].shape[2]

    # pre-allocate memory to store the losses
    avg_train_loss_per_epoch = []
    train_loss_to_write = []
    if validation_data:
        avg_validation_loss_per_epoch = []
        validation_loss_to_write = []

    # define the type of semantics bloks used
    if semantics:
        semantic_blocks = ["cut","blocks","pizza","random"]
    else:
        semantic_blocks = None

    # define the sampling threshold to use if max_percentage_context is None
    threshold = 1/3

    # define the optimizer
    opt = torch.optim.Adam(model.parameters(),
                           learning_rate,
                           weight_decay=weight_decay)

    for i in range(epochs):

        print("Epoch:", i+epoch_start+1)
        model.train()
        train_losses = []
        iterator = tqdm(train_data)
        for batch_idx, (data, target) in enumerate(iterator):

            # TODO: remove this (debugging)
            #if batch_idx == 1:
                #x = data
                #y = target
            #data = x
            #target = y
            # either select nbr of context pts between 2 and max_percentage_context, or uniformly between 2 and 1/3 with probability 1/2 and between 1/3 and 1 with probability 1/2
            if max_percentage_context:
                num_context_points = np.random.randint(min_context_points,int(img_height * img_width * max_percentage_context))
            else:
                s = np.random.rand()
                if s < 1/2:
                    num_context_points = np.random.randint(min_context_points,int(img_height * img_width * threshold))
                else:
                    num_context_points = np.random.randint(int(img_height * img_width * threshold), img_height * img_width)

            if convolutional:
                mask, context_img = image_processor(data, num_context_points, convolutional=convolutional, semantic_blocks=semantic_blocks, device=device)
                data = data.to(device).permute(0,2,3,1)
                train_loss = model.train_step(mask,context_img,data, opt)
            else:
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional, semantic_blocks=semantic_blocks, device=device)
                train_loss = model.train_step(x_context,y_context,x_target, y_target, opt)
            # store the loss
            train_losses.append(train_loss)

            if batch_idx == 0 or (batch_idx + 1) % report_freq == 0: # report the loss
                avg_train_loss = np.array(train_losses).mean()
                txt = report_loss("Training", avg_train_loss, batch_idx)
                iterator.set_description(txt)
                iterator.refresh()  # to show immediately the update

        # compute the average loss over the epoch and store all losses
        epoch_avg_train_loss = np.array(train_losses).mean()
        avg_train_loss_per_epoch.append(epoch_avg_train_loss)
        train_loss_to_write.append(epoch_avg_train_loss)

        # Print the average epoch loss
        print("Average training loss:", epoch_avg_train_loss)

        # calculate the validation loss
        if validation_data:
            validation_losses = []
            for batch_idx, (data, target) in enumerate(validation_data):
                if max_percentage_context:
                    num_context_points = np.random.randint(min_context_points,
                                                           int(img_height * img_width * max_percentage_context))
                else:
                    s = np.random.rand()
                    if s < 1 / 2:
                        num_context_points = np.random.randint(min_context_points,
                                                               int(img_height * img_width * threshold))
                    else:
                        num_context_points = np.random.randint(int(img_height * img_width * threshold),
                                                               img_height * img_width)
                if convolutional:
                    mask, context_img = image_processor(data, num_context_points, convolutional, semantic_blocks=semantic_blocks, device = device)
                    mean, std = model(mask,context_img)
                    data = data.to(device).permute(0,2,3,1)
                    validation_loss = model.loss(mean,std,data).item()
                else:
                    x_context, y_context, x_target, y_target = image_processor(data, num_context_points,
                                                                               convolutional, semantic_blocks=semantic_blocks,
                                                                               device = device)
                    mean, std = model(x_context, y_context, x_target)
                    validation_loss = model.loss(mean, std, y_target).item()
                validation_losses.append(validation_loss)
            epoch_avg_validation_loss = np.array(validation_losses).mean()
            avg_validation_loss_per_epoch.append(epoch_avg_validation_loss)
            validation_loss_to_write.append(epoch_avg_validation_loss)

            # Print the average epoch loss
            print("Average validation loss:", epoch_avg_validation_loss)

        # save the checkpoint
        if save_freq and (i+1) % save_freq == 0:
            # update the file name
            save_dir = model_save_dir.copy()
            save_dir[-3] = str(epoch_start + i+1)
            save_dir = "".join(save_dir)

            # save the model
            torch.save(model.state_dict(), save_dir)

            # write the average epoch loss to the txt file
            with open(train_loss_dir_txt, 'a+') as f:
                for loss_item in train_loss_to_write:
                    f.write('%s\n' % loss_item)
            train_loss_to_write = []

            # write the average epoch validation loss to the txt file if some validation data is supplied
            if validation_data:
                with open(validation_loss_dir_txt, 'a+') as f:
                    for loss_item in validation_loss_to_write:
                        f.write('%s\n' % loss_item)
                validation_loss_to_write = []

                if visualisation_dir:
                    # get the output directory
                    for num_context_points in [10,100,img_height * img_width]:
                        visualisation_dir = visualisation_dir.copy()
                        visualisation_dir[5] = str(epoch_start + i + 1)
                        visualisation_dir[-2] = str(num_context_points)
                        img_output_dir = "".join(visualisation_dir)

                        # create directories if it doesn't exist yet
                        dir_to_create = "".join(visualisation_dir[:3])
                        os.makedirs(dir_to_create, exist_ok=True)
                        if num_context_points == img_height * img_width:
                            qualitative_evaluation_images(model, validation_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir, convolutional=convolutional,
                                                          semantic_blocks=["random"])
                        else:
                            qualitative_evaluation_images(model, validation_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir, convolutional=convolutional,
                                                          semantic_blocks=semantic_blocks)

    # write the final losses
    with open(train_loss_dir_txt, 'a+') as f:
        for loss_item in train_loss_to_write:
            f.write('%s\n' % loss_item)
    if validation_data:
        with open(validation_loss_dir_txt, 'a+') as f:
            for loss_item in validation_loss_to_write:
                f.write('%s\n' % loss_item)

    return avg_train_loss_per_epoch, avg_validation_loss_per_epoch

def train_sup(train_data,model,epochs, model_save_dir, train_loss_dir_txt, validation_data = None, validation_loss_dir_txt = "", convolutional=False, augment_missing = False, is_CNP = True, report_freq = 100, learning_rate=1e-3, weight_decay=1e-5, save_freq = 10, n_best_checkpoint = None, epoch_start = 0, device=torch.device('cpu')):
    min_percentage_missing_pixels = 0.75

    img_height, img_width = train_data.dataset[0][0].shape[1], train_data.dataset[0][0].shape[2]

    # pre-allocate memory to store the losses
    avg_train_loss_per_epoch = []
    avg_validation_loss_per_epoch = []
    train_loss_to_write = []
    validation_loss_to_write = []

    # define the optimizer
    opt = torch.optim.Adam(model.parameters(),
                           learning_rate,
                           weight_decay=weight_decay)

    for i in range(epochs):
        print("Epoch:", i + epoch_start + 1)
        model.train()
        train_losses = []
        iterator = tqdm(train_data)
        for batch_idx, (data, target) in enumerate(iterator):
            target = target.to(device)
            if augment_missing:
                num_context_points = np.random.randint(int(min_percentage_missing_pixels * img_height * img_width),
                                                       int(img_height * img_width))
            else:
                num_context_points = img_height * img_width
            if is_CNP:
                if convolutional:
                    mask, context_img = image_processor(data, num_context_points, convolutional=convolutional,
                                                        device=device)
                    data = data.to(device)
                    loss, accuracy, total = model.train_step(mask, context_img, target, opt)
                else:
                    x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional=convolutional,
                                                                               device=device)
                    loss = model.train_step(x_context, y_context, target, opt)
            else:
                data = data.to(device)
                loss = model.train_step(data,target,opt)

            # store the loss
            train_losses.append(loss)

            if batch_idx == 0 or (batch_idx + 1) % report_freq == 0:  # report the loss
                avg_train_loss = np.array(train_losses).mean()
                txt = report_loss("Training", avg_train_loss, batch_idx)
                iterator.set_description(txt)
                iterator.refresh()  # to show immediately the update

        # compute the average loss over the epoch and store all losses
        epoch_avg_train_loss = np.array(train_losses).mean()
        avg_train_loss_per_epoch.append(epoch_avg_train_loss)
        train_loss_to_write.append(epoch_avg_train_loss)

        # Print the average epoch loss
        print("Average training loss:", epoch_avg_train_loss)

        # calculate the validation loss
        if validation_data:
            validation_losses = []
            for batch_idx, (data, target) in enumerate(validation_data):
                target = target.to(device)
                if is_CNP:
                    if convolutional:
                        mask, context_img = image_processor(data, num_context_points, convolutional=convolutional,
                                                                                   device=device)
                        data = data.to(device)
                        output_score, output_logit = model(mask,context_img)
                    else:
                        x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional=convolutional,
                                                                                   device=device)
                        output_score, output_logit = model(x_context,y_context)
                else:
                    data = data.to(device)
                    output_score, output_logit = model(data)
                validation_losses.append(model.loss(output_score,target).item())
            epoch_avg_validation_loss = np.array(validation_losses).mean()
            avg_validation_loss_per_epoch.append(epoch_avg_validation_loss)
            validation_loss_to_write.append(epoch_avg_validation_loss)

            # Print the average epoch loss
            print("Average validation loss:", epoch_avg_validation_loss)


        # save the checkpoint and losses
        if (i + 1) % save_freq == 0:
            # update the file name
            save_dir = model_save_dir.copy()
            save_dir[-3] = str(epoch_start + i + 1)
            save_dir = "".join(save_dir)

            # save the model
            if not (n_best_checkpoint):
                torch.save(model.state_dict(), save_dir)

            # write the average epoch train loss to the txt file
            with open(train_loss_dir_txt, 'a+') as f:
                for loss_item in train_loss_to_write:
                    f.write('%s\n' % loss_item)
            train_loss_to_write = []

            # write the average epoch validation loss to the txt file if some validation data is supplied
            if validation_data:
                with open(validation_loss_dir_txt, 'a+') as f:
                    for loss_item in validation_loss_to_write:
                        f.write('%s\n' % loss_item)
                validation_loss_to_write = []

        if n_best_checkpoint:
            pass
            # TODO: n-best checkpoint saving


    # write the final losses
    with open(train_loss_dir_txt, 'a+') as f:
        for loss_item in train_loss_to_write:
            f.write('%s\n' % loss_item)
    if validation_data:
        with open(validation_loss_dir_txt, 'a+') as f:
            for loss_item in validation_loss_to_write:
                f.write('%s\n' % loss_item)

    return avg_train_loss_per_epoch, avg_validation_loss_per_epoch


def train_joint(train_data,model,epochs, model_save_dir, train_joint_loss_dir_txt, train_unsup_loss_dir_txt, train_accuracy_dir_txt, validation_data = None, validation_joint_loss_dir_txt = "", validation_unsup_loss_dir_txt = "", validation_accuracy_dir_txt = "", visualisation_dir = None, semantics=False, convolutional=False, variational=False, min_context_points = 2, report_freq = 100, learning_rate=1e-3, weight_decay=1e-5, save_freq = 10, n_best_checkpoint = None, epoch_start = 0, device=torch.device('cpu'), alpha=None, alpha_validation=None, num_samples_expectation=None, std_y=None, parallel=False, weight_ratio=False, consistency_regularization=False):

    img_height, img_width = train_data.dataset[0][0].shape[1], train_data.dataset[0][0].shape[2]

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

    # define the type of semantics bloks used
    if semantics:
        semantic_blocks = ["cut", "blocks", "pizza", "random"]
    else:
        semantic_blocks = None

    # use more sets of context sets if using consistency regularization
    if consistency_regularization:
        num_sets_of_context = 2
    else:
        num_sets_of_context = 1

    # define the sampling threshold to use if max_percentage_context is None
    threshold = 1 / 3

    for i in range(epochs):
            
        print("Epoch:", i + epoch_start + 1)
        model.train()
        train_losses = [[],[]]
        train_num_correct  = []
        train_totals = []
        iterator = tqdm(train_data)
        for batch_idx, (data, target) in enumerate(iterator):

            target = target.to(device)

            if consistency_regularization:
                repeat_size_target = [1] * len(target.shape)
                repeat_size_target[0] = num_sets_of_context
                repeat_size_target = tuple(repeat_size_target)
                target = target.repeat(repeat_size_target)
                repeat_size_data = [1] * len(data.shape)
                repeat_size_data[0] = num_sets_of_context
                repeat_size_data = tuple(repeat_size_data)
                data = data.repeat(repeat_size_data)

            # TODO: remove this (debugging)
            #if batch_idx == 0 and i == 0:
            #    x = data
            #    y = target
            #data = x
            #target = y

            s = np.random.rand()
            if s < 1 / 2:
                num_context_points = np.random.randint(min_context_points, int(img_height * img_width * threshold))
            else:
                num_context_points = np.random.randint(int(img_height * img_width * threshold), img_height * img_width)
            
            # TODO: remove this (debugging):
            #num_context_points = 784
            
            # TODO: add variational conv
            if convolutional:
                mask, context_img, num_context_points, num_target_points = image_processor(data, num_context_points, convolutional=convolutional,
                                                                                           semantic_blocks=semantic_blocks, device=device, return_num_points=True)

                if weight_ratio:
                    scale_sup = num_context_points/num_target_points
                    scale_unsup = 1-scale_sup
                else:
                    scale_sup, scale_unsup = 1,1

                data = data.to(device)
                train_joint_loss, train_sup_loss, train_unsup_loss, accuracy, total = model.joint_train_step(mask, context_img, target, data, opt, alpha=alpha, scale_sup=scale_sup, scale_unsup=scale_unsup, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)
            else:
                x_context, y_context, x_target, y_target, num_context_points, num_target_points = image_processor(data, num_context_points,
                                                                                                                  convolutional=convolutional,
                                                                                                                  semantic_blocks=semantic_blocks,
                                                                                                                  device=device, return_num_points=True)
                if weight_ratio:
                    scale_sup = num_context_points/num_target_points
                    scale_unsup = 1-scale_sup
                else:
                    scale_sup, scale_unsup = 1,

                if not(variational):
                    train_joint_loss, train_sup_loss, train_unsup_loss, accuracy, total = model.joint_train_step(x_context, y_context, x_target, target, y_target, opt, alpha=alpha, scale_sup=scale_sup, scale_unsup=scale_unsup, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)
                else:
                    train_joint_loss, train_sup_loss, train_unsup_loss, accuracy, total = model.joint_train_step(x_context,y_context,x_target,target,y_target,opt,alpha,num_samples_expectation=num_samples_expectation, std_y=std_y, parallel=parallel, scale_sup=scale_sup, scale_unsup=scale_unsup, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)

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

                # TODO: remove this (debugging)
                #data = x
                #target = y

                target = target.to(device)

                if consistency_regularization:
                    repeat_size_target = [1] * len(target.shape)
                    repeat_size_target[0] = num_sets_of_context
                    repeat_size_target = tuple(repeat_size_target)
                    target = target.repeat(repeat_size_target)
                    repeat_size_data = [1] * len(data.shape)
                    repeat_size_data[0] = num_sets_of_context
                    repeat_size_data = tuple(repeat_size_data)
                    data = data.repeat(repeat_size_data)
                
                s = np.random.rand()
                if s < 1 / 2:
                    num_context_points = np.random.randint(min_context_points, int(img_height * img_width * threshold))
                else:
                    num_context_points = np.random.randint(int(img_height * img_width * threshold), img_height * img_width)

                # TODO: add variational conv
                if convolutional:
                    mask, context_img = image_processor(data, num_context_points, convolutional=convolutional,
                                                        semantic_blocks=semantic_blocks, device=device)
                    data = data.to(device)

                    # get the losses
                    joint_loss, sup_loss, unsup_loss, accuracy, total = model.joint_loss(mask,context_img,target,data,alpha=alpha_validation, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)
                    joint_loss = joint_loss.item()
                else:
                    x_context, y_context, x_target, y_target = image_processor(data, num_context_points,
                                                                               convolutional=convolutional,
                                                                               semantic_blocks=semantic_blocks,
                                                                               device=device)
                    if not(variational):
                        joint_loss, sup_loss, unsup_loss, accuracy, total = model.joint_loss(x_context, y_context, x_target, target, y_target, alpha=alpha_validation, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)
                        unsup_loss = joint_loss.item()
                    else:
                        obj, sup_loss, unsup_loss, accuracy, total = model.joint_loss(x_context,y_context,x_target,target,y_target,alpha_validation,num_samples_expectation,std_y, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)
                        joint_loss = obj.item()
                num_correct = accuracy * total
        
                validation_losses[0].append(joint_loss)
                validation_losses[1].append(unsup_loss)
                validation_num_correct.append(num_correct)
                validation_totals.append(total)

            # TODO: remove this (debugging)
            """
            mean = model(x_context,y_context, x_target)
            mean = mean.detach().cpu().numpy().reshape((-1, img_width,img_height,1))
            plt.figure()
            plt.imshow(mean[0])
            plt.savefig("tmp/tmp_mean" + str(i) + ".svg")
            plt.figure()
            context_img = format_context_points_image(x_context,y_context,img_height,img_width)
            plt.imshow(context_img[0])
            plt.savefig("tmp/tmp_ctxt" + str(i) + ".svg")
            """

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
                    for num_context_points in [10,100,img_height * img_width]:
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

                        if num_context_points == img_height * img_width:
                            qualitative_evaluation_images(model, train_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir_train, convolutional=convolutional,
                                                          semantic_blocks=["random"], variational=variational,
                                                          include_class_predictions=True)
                        else:
                            qualitative_evaluation_images(model, train_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir_train, convolutional=convolutional,
                                                          semantic_blocks=semantic_blocks, variational=variational,
                                                          include_class_predictions=True)

                        # validation data
                        visualisation_dir_validation = visualisation_dir.copy()
                        visualisation_dir_validation[2] += "validation/"
                        img_output_dir_validation = "".join(visualisation_dir_validation)

                        # create directories if it doesn't exist yet
                        dir_to_create = "".join(visualisation_dir_validation[:3])
                        os.makedirs(dir_to_create, exist_ok=True)

                        if num_context_points == img_height * img_width:
                            qualitative_evaluation_images(model, validation_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir_validation, convolutional=convolutional,
                                                          semantic_blocks=["random"], variational=variational,
                                                          include_class_predictions=True)
                        else:
                            qualitative_evaluation_images(model, validation_data, num_context_points=num_context_points,
                                                          device=device,
                                                          save_dir=img_output_dir_validation, convolutional=convolutional,
                                                          semantic_blocks=semantic_blocks, variational=variational,
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


