from Utils.data_processor import image_processor, format_context_points_image, context_points_image_from_mask
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch
import math
import random

def test_model_accuracy(model,test_data,device,convolutional=False,num_context_points=784,is_CNP=True):
    sum, total = 0,0
    for i,(data,target) in enumerate(test_data):
        target = target.to(device)
        if is_CNP:
            if convolutional:
                mask, context_img = image_processor(data, num_context_points, convolutional=convolutional, device=device)
                data = data.to(device)
                batch_accuracy, batch_size = model.evaluate_accuracy(mask,context_img,target)
            else:
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional=convolutional, device=device)
                batch_accuracy, batch_size = model.evaluate_accuracy(x_context,y_context,target)
        else:
            data = data.to(device)
            batch_accuracy, batch_size = model.evaluate_accuracy(data, target)
        sum += batch_accuracy * batch_size
        total += batch_size
    return sum/total

def test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=False,num_context_points=784, save_freq=20, is_CNP=False, best="min"):

    # get the optimal epoch number
    N = 10 # window size for the smoothing the loss (moving average)
    epoch = find_optimal_epoch_number(validation_loss_dir_txt, save_freq=save_freq, window_size=N, best=best)

    # load the corresponding model
    load_dir = model_save_dir.copy()
    load_dir[-3] = str(epoch)
    load_dir = "".join(load_dir)

    # load the model
    model.load_state_dict(torch.load(load_dir, map_location=device))

    # get the accuracy
    accuracy = test_model_accuracy(model, test_data, device, convolutional=convolutional, is_CNP=is_CNP)

    return accuracy


def test_model_accuracy_all_epochs(model,model_save_dir_base,data,device,convolutional=False,is_CNP=True):

    accuracies = []
    epochs = []
    # load the corresponding model
    for load_dir in os.listdir(model_save_dir_base):

        if ".pth" in load_dir:
            # load the weights
            model.load_state_dict(torch.load(model_save_dir_base + load_dir, map_location=device))

            # find the epoch number
            epoch_num = float(load_dir.split("-")[1].split("E")[0])
            epochs.append(epoch_num)

            # get the accuracy
            accuracy = test_model_accuracy(model, data, device, convolutional=convolutional, is_CNP=is_CNP)
            accuracies.append(accuracy)

    return accuracies, epochs


def evaluate_model_full_accuracy(model,model_save_dir_base,acc_dir_txt,data,device,convolutional=False,is_CNP=True):
    accuracies, epochs = test_model_accuracy_all_epochs(model,model_save_dir_base,data,device,
                                                        convolutional=convolutional,is_CNP=is_CNP)

    with open(acc_dir_txt, "w") as f:
        for i,acc in enumerate(accuracies):
            epoch = epochs[i]
            txt = str(epoch) + ", " + str(acc) + "\n"
            f.write(txt)





def find_optimal_epoch_number(validation_loss_dir_txt, save_freq=20, window_size=10, best="min"):

    assert best in ["min","max"], "Argument best should be one of [min,max] but was given: " + str(best)

    # read in the validation loss
    l = len(validation_loss_dir_txt)
    losses = []
    with open(validation_loss_dir_txt, "r") as f:
        for x in f.read().split():
            if x != "":
                losses.append(float(x))

    # smooth the losses with a moving average (code from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean)
    N = window_size
    cumsum, moving_aves = [0], []
    for i, x in enumerate(losses, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
        elif i < N:
            moving_aves.append(cumsum[i] / (i + 1))

    # get the index of the minimum
    if best == "min":
        _, idx = min((val, idx) for (idx, val) in enumerate(moving_aves))
    elif best == "max":
        _, idx = max((val, idx) for (idx, val) in enumerate(moving_aves))

    # get the closest corresponding epoch number for which a checkpoint was saved
    epoch = int(round((idx + 1) / save_freq) * save_freq)
    epoch = max(epoch,save_freq) # ensure that it does not return epoch 0, at least take the first checkpoint

    return epoch


def plot_loss(list_loss_dir_txt,loss_dir_plot,labels=None,y_label="Loss",styles=None,ax=None):
    l = len(list_loss_dir_txt)
    losses = [[] for _ in range(l)]
    for i,filename in enumerate(list_loss_dir_txt):
        with open(filename,"r") as f:
            for x in f.read().split():
                if x != "":
                    losses[i].append(float(x))

    # plot on given figure if passed as input (to allow using as subplots)
    if ax == None:
        ax_given = False
        fig, ax = plt.subplots(1)

    else:
        ax_given = True


    for i in range(l):
        if styles != None:
            ax.plot(np.arange(1, len(losses[i]) + 1), losses[i], styles[i])
        else:
            ax.plot(np.arange(1, len(losses[i]) + 1), losses[i])
    if labels == None:
        if l == 2:
            ax.legend(["Train loss", "Validation loss"])
    else:
        ax.legend(labels)

    ax.set_xlabel("Epoch",fontsize=15)
    ax.set_ylabel(y_label,fontsize=15)
    if y_label != None and "accuracy" in y_label.lower():
        ax.set_ylim([0,1])

    if not(ax_given):
        plt.savefig(loss_dir_plot)
        plt.close(fig)

def qualitative_evaluation_images(model, data, num_context_points, device, save_dir, convolutional=False, semantic_blocks=None, variational=False, include_class_predictions=False):

    # number of images to show per class
    if include_class_predictions:
        num_img_per_class = 5 # true image, context image, predicted mean, predicted std, class predictions
    else:
        num_img_per_class = 4 # true image, context image, predicted mean, predicted std

    # get image height and width
    num_channels, img_height, img_width= data.dataset[0][0].shape[0], data.dataset[0][0].shape[1], data.dataset[0][0].shape[2]

    # get the data to plot (one for every class)
    try:
        num_classes = max(data.dataset.dataset.targets).item() + 1
    except AttributeError:
        list_targets = [x[1] for x in data.dataset]
        num_classes = max(list_targets) + 1

    # get one data image per class
    selected_classes = [False for _ in range(num_classes)]
    images_to_plot = [None for _ in range(num_classes)]
    found = 0
    quit = False
    for images, labels in data:
        for i in range(images.shape[0]):
            image = images[i]
            label = labels[i]
            if not selected_classes[label]:
                images_to_plot[label] = (image,label)
                found += 1
                selected_classes[label] = True
            if found >= num_classes:
                break
                quit = True
        if quit:
            break

    # num of columns and rows in the subplot
    num_cols = num_classes
    num_rows = num_img_per_class * math.ceil(num_classes/num_cols)

    # use subplots to visualize all images together
    fig, ax = plt.subplots(num_rows,num_cols,figsize=(num_cols,num_rows))
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    # initialize the row and column position in the subplot
    row = 0
    col = 0

    for (image,label) in images_to_plot:
        data = image.unsqueeze(0) # add the batch dimension
        label = label.unsqueeze(0)
        if convolutional:
            mask, context_img = image_processor(data, num_context_points, convolutional, semantic_blocks=semantic_blocks, device=device)
            if not(variational):
                if not(model.is_gmm):
                    if include_class_predictions:
                        if model.classify_same_image:
                            logits, probs, mean, std, probs_same_image = model(mask,context_img,joint=True)
                        else:
                            logits, probs, mean, std = model(mask,context_img,joint=True)
                        probabilities_to_plot = probs[0].detach().cpu()
                    else:
                        mean, std = model(mask,context_img)
                    mean = mean.detach().cpu().numpy()
                    std = std.detach().cpu().numpy()
                    img1, img2 = mean, std
                else:
                    mean, std, probs, samples = model.sample_one_component(mask,context_img)
                    class_sampled = samples[0].item()
                    probabilities_to_plot = probs[0].detach().cpu()
                    mean = mean.detach().cpu().numpy()
                    std = std.detach().cpu().numpy()
                    img1, img2 = mean, std

            else:
                sample1, probs = model(mask,context_img).detach().cpu().numpy()
                sample2, probs = model(mask,context_img).detach().cpu().numpy()
                probabilities_to_plot = probs[0].detach().cpu()
                img1, img2 = sample1, sample2
            context_img = context_points_image_from_mask(mask, context_img)
        else:
            x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional, semantic_blocks=semantic_blocks, device=device)
            if not(variational):
                if not(model.is_gmm):
                    mean,std = model(x_context,y_context,x_target)
                    mean = mean.detach().cpu().numpy().reshape((-1, img_width,img_height,num_channels))
                    std = std.detach().cpu().numpy().reshape((-1, img_width, img_height, num_channels))
                    img1, img2 = mean, std
                else:
                    mean, std, probs, samples = model.sample_one_component(x_context,y_context,x_target)
                    class_sampled = samples[0].item()
                    probabilities_to_plot = probs[0].detach().cpu()
                    mean = mean.detach().cpu().numpy().reshape((-1, img_width,img_height,num_channels))
                    std = std.detach().cpu().numpy().reshape((-1, img_width, img_height, num_channels))
                    img1, img2 = mean, std
            else:
                sample1, probs = model(x_context,y_context,x_target).detach().cpu().numpy().reshape((-1, img_width,img_height,num_channels))
                sample2, probs = model(x_context,y_context,x_target).detach().cpu().numpy().reshape((-1, img_width,img_height,num_channels))
                probabilities_to_plot = probs[0].detach().cpu()
                img1, img2 = sample1, sample2
            context_img = format_context_points_image(x_context,y_context,img_height,img_width)


        ax[row,col].imshow(data[0].permute(1,2,0).detach().cpu().numpy())
        ax[row+1,col].imshow(context_img[0])
        ax[row+2,col].imshow(img1[0]) # mean for CNP and sample 1 for NP
        ax[row+3,col].imshow(img2[0]) # std for CNP and sample 1 for NP

        if include_class_predictions:
            # get the colors of the bars (red for the one we sampled from)
            colors_bar = ["blue" for _ in range(num_classes)]
            if model.is_gmm:
                colors_bar[class_sampled] = "red"

            ax[row+4, col].bar(range(num_classes),probabilities_to_plot,color=colors_bar)
            ax[row+4, col].set_xticks(np.arange(num_classes))
            ax[row + 4, col].set_xticklabels(np.arange(num_classes), fontsize=5)
            ax[row + 4, col].set_yticks([])
            ax[row + 4, col].set_ylim(0,1)


        row += num_img_per_class

        if row >= num_rows:
            row = 0
            col += 1

    # remove the axes
    for idx1 in range(num_rows-1):
        for idx2 in range(num_cols):
            ax[idx1,idx2].set_axis_off()

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()

def qualitative_evaluation_GP(model, data, num_context_points, num_test_points=100, device=torch.device('cpu'), save_dir="", variational=False, num_samples_variational=50, include_class_predictions=False):

    if include_class_predictions:
        num_img_per_class = 2
    else:
        num_img_per_class = 1

    transparency = 0.1
    num_classes = data.num_kernels

    # num of columns and rows in the subplot
    num_cols = num_classes #math.ceil(num_classes/2)
    num_rows = num_img_per_class * math.ceil(num_classes / num_cols)

    # use subplots to visualize all images together
    fig, ax = plt.subplots(num_rows,num_cols,figsize=(num_cols*5,num_rows*5))
    if num_rows == 1:
        ax = ax[None,:]
    elif num_cols == 1:
        ax = ax[:,None]
    #plt.subplots_adjust(wspace=0.01,hspace=0.01)

    # initialize the row and column position in the subplot
    row = 0
    col = 0

    for class_idx in range(num_classes):
        task, label = data.generate_task(class_idx,num_context_points,num_test_points)
        x_context = torch.unsqueeze(task["x_context"][0],dim=0)
        y_context = torch.unsqueeze(task["y_context"][0],dim=0)
        x_target = torch.unsqueeze(task["x"][0],dim=0)
        y_target = torch.unsqueeze(task["y"][0],dim=0)

        # plot the context points and the true function
        ax[row, col].plot(x_context[0,:,0].cpu(), y_context[0,:,0].cpu(), 'k.')
        ax[row, col].plot(x_target[0,:,0].cpu(), y_target[0,:,0].cpu(), 'r--')

        if variational:
            for i in range(num_samples_variational):
                y_prediction, std, probs = model(x_context,y_context,x_target)
                y_prediction = y_prediction[0,:,0].detach().cpu()
                probabilities_to_plot = probs[0].detach().cpu()
                ax[row, col].plot(x_target[0,:,0].cpu(),y_prediction,'b-',alpha=transparency)
        else:
            if model.is_gmm:
                mean, std, logits, probs = model(x_context,y_context,x_target)
                probabilities_to_plot = probs[0].detach().cpu()
                for j in range(num_classes):
                    weight = probs[0,j].item()
                    mean_component, std_component = mean[0, j, :, 0].detach().cpu(), std[0, j, :, 0].detach().cpu()
                    ax[row,col].plot(x_target[0,:,0].cpu(),mean_component,'b-',alpha=weight)
                    ax[row,col].fill_between(x_target[0,:,0].cpu(), mean_component - 1.96 * std_component, mean_component + 1.96 * std_component,alpha=0.5*weight, color='b')
            else:
                mean, std = model(x_context,y_context,x_target)
                mean, std = mean[0,:,0].detach().cpu(), std[0,:,0].detach().cpu()
                ax[row, col].plot(x_target[0,:,0].cpu(), mean, 'b-')
                ax[row, col].fill_between(x_target[0,:,0].cpu(), mean - 1.96 * std, mean + 1.96 * std, alpha=0.5 * weight, color='b')
        if include_class_predictions:

            # true kernel probabilities 
            logpdfs = np.array([f(x_context.detach().cpu().numpy()[0,:,0],data.noise).logpdf(y_context.detach().cpu().numpy()[0,:,0]) for f in data.gps])
            logpdfs -= np.max(logpdfs)
            true_probs = np.exp(logpdfs) / np.sum(np.exp(logpdfs))

            if data.kernel_names:
                kernel_labels = data.kernel_names
            else:
                kernel_labels = list(range(num_classes))
            ax[row+1, col].bar(range(num_classes),probabilities_to_plot,alpha=0.5,color="blue", label="Predicted probability")
            ax[row+1, col].bar(range(num_classes),true_probs,alpha=0.5,color="red",label="True GP probability")
            ax[row+1,col].legend(fontsize="x-large")
            ax[row+1, col].set_xticks(np.arange(num_classes))
            ax[row+1, col].set_xticklabels(kernel_labels,rotation=45)
            ax[row+1, col].set_xlabel("Kernel",fontsize=15)
            ax[row+1, col].set_ylabel("Probability",fontsize=15)
            ax[row+1, col].set_ylim(0,1)

        row += num_img_per_class
        if row >= num_rows:
            row = 0
            col += 1

    #plt.tight_layout()
    plt.savefig(save_dir)

class LossWriter():
    """ Object to store the losses after every epochs and write them to the loss file

    Args:
        dir_txt (list of str): directory where to store the losses (dir_txt[1] is where to insert the loss name)
    """
    def __init__(self,list_dir_txt):

        self.list_losses_during_epoch = {}
        self.list_avg_losses_per_epoch = {}
        self.list_losses_to_write = {}
        self.list_dir_txt = list_dir_txt

    def append_losses_during_epoch(self,dic_losses):
        for (key,value) in dic_losses.items():
            if not key in self.list_losses_during_epoch:
                self.list_losses_during_epoch[key] = []
            self.list_losses_during_epoch[key].append(value)

    def obtain_avg_losses_current_epoch(self,list_losses_name):
        out = []
        for loss_name in list_losses_name:
            out.append(self.compute_epoch_avg_loss(loss_name))
        return out

    def compute_epoch_avg_loss(self,loss_name):
        if "accuracy" in loss_name:
            suffix = loss_name.split("accuracy")[1]
            accuracies = np.array(self.list_losses_during_epoch[loss_name])
            total_name = "total" + suffix
            totals = np.array(self.list_losses_during_epoch[total_name])
            num_correct = accuracies * totals
            accuracy = np.sum(num_correct) / np.sum(totals)
            return accuracy
        elif "total" in loss_name:
            sum_total = sum(self.list_losses_during_epoch[loss_name])
            return sum_total
        else:
            array_losses = np.array(self.list_losses_during_epoch[loss_name])
            avg_loss = np.nanmean(array_losses)
            return avg_loss

    def append_epoch_avg_losses(self):
        for (key,value) in self.list_losses_during_epoch.items():
            if not key in self.list_avg_losses_per_epoch:
                self.list_avg_losses_per_epoch[key] = []
                self.list_losses_to_write[key] = []
            avg_loss = self.compute_epoch_avg_loss(key)
            self.list_avg_losses_per_epoch[key].append(avg_loss)
            self.list_losses_to_write[key].append(avg_loss)
            self.list_losses_during_epoch[key] = []

    def write_losses(self):
        
        for (key,list_losses) in self.list_losses_to_write.items():

            local_loss_dir_txt = self.obtain_loss_dir_txt(key)

            # write the loss
            with open(local_loss_dir_txt, 'a+') as f:
                for i, value in enumerate(list_losses):
                    with open(local_loss_dir_txt, 'a+') as f:
                        f.write('%s\n' % value)

            self.list_losses_to_write[key] = []

    def obtain_loss_dir_txt(self,loss_name):

        # get the path to the right loss file
        list_loss_dir_txt = self.list_dir_txt.copy()
        list_loss_dir_txt[1] = loss_name
        local_loss_dir_txt = "".join(list_loss_dir_txt)

        # create the directory if necessary
        dir_to_create = os.path.dirname("".join(local_loss_dir_txt))
        os.makedirs(dir_to_create, exist_ok=True)

        return local_loss_dir_txt

def plot_losses_from_loss_writer(train_loss_writer,validation_loss_writer=None):
    """ Plot the losses given two LossWriter objects

    Args:
        train_loss_writer (LossWriter): LossWriter object containing the train losses
        validation_loss_writer (LossWriter), optional: LossWriter object containing the validation losses

    """

    list_loss_dir_txt_subplots = []
    y_label_subplots = []

    for (key,value) in train_loss_writer.list_avg_losses_per_epoch.items():

        list_loss_dir_txt = [train_loss_writer.obtain_loss_dir_txt(key)]
        if (validation_loss_writer != None) and (key in validation_loss_writer.list_avg_losses_per_epoch):
            list_loss_dir_txt.append(validation_loss_writer.obtain_loss_dir_txt(key))

        plot_dir = "".join(list_loss_dir_txt[0].split("train"))
        plot_dir = "_".join(plot_dir.split("__")) # remove double underscores
        plot_dir = plot_dir.split(".txt")[0] + ".svg"

        if "accuracy" in key.lower():
            y_label = "Accuracy"
        else:
            y_label = None

        plot_loss(list_loss_dir_txt,plot_dir,y_label=y_label)

        # keep the four losses to plot them in the subplot
        if key in ["rec_loss", "cons_loss", "accuracy_discriminator", "accuracy"]:
            list_loss_dir_txt_subplots.append(list_loss_dir_txt)
            y_label_subplots.append(y_label)

    # Make the subplot
    num_cols = 4
    num_plots = len(list_loss_dir_txt_subplots)
    num_rows = math.ceil(num_plots / num_cols)

    plot_dir = plot_dir.split("average")[0] + "overview.svg"

    fig, ax = plt.subplots(num_rows, min(num_cols,num_plots), figsize=(num_cols * 5, num_rows * 5))
    if num_rows == 1:
        ax = ax[None, :]
    elif num_cols == 1:
        ax = ax[:, None]

    current_col = -1
    current_row = 0
    for i,loss_dir in enumerate(list_loss_dir_txt_subplots):
        current_col += 1
        if current_col > num_cols-1:
            current_row += 1
            current_col = 0
        plot_loss(loss_dir, plot_dir, y_label=y_label_subplots[i],ax=ax[current_row,current_col])

    fig.savefig(plot_dir)

class InfoWriter():
    """Class to store information regarding the training of a network

    Args:
        info_dir_txt (str): path to the file containing the information
    """

    def __init__(self,info_dir_txt):
        self.info_dir_txt = info_dir_txt
        self.info = {}

    def read_file(self):
        with open(self.info_dir_txt, 'r') as f:
            self.info = json.load(f)

    def write_to_file(self):
        with open(self.info_dir_txt, 'w') as f:
            f.write(json.dumps(self.info))

    def update_time(self,t):

        # create the directory if necessary
        dir_to_create = os.path.dirname(self.info_dir_txt)
        os.makedirs(dir_to_create, exist_ok=True)

        # make sure to have the latest version of the information in the dictionary
        if os.path.isfile(self.info_dir_txt):
            self.read_file()

        # update/store the training time
        if "training_time" in self.info:
            self.info["training_time"] += t
        else:
            self.info["training_time"] = t

        self.write_to_file()

    def update_number_of_parameters(self,n):

        # create the directory if necessary
        dir_to_create = os.path.dirname(self.info_dir_txt)
        os.makedirs(dir_to_create, exist_ok=True)

        # make sure to have the latest version of the information in the dictionary
        if os.path.isfile(self.info_dir_txt):
            self.read_file()

        self.info["num_params"] = n

        self.write_to_file()












