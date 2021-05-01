from Utils.data_processor import image_processor
import matplotlib.pyplot as plt
import numpy as np
import torch

def test_model_accuracy(model,test_data,device,convolutional=False,num_context_points=784,is_CNP=True):
    sum, total = 0,0
    for i,(data,target) in enumerate(test_data):
        target = target.to(device)
        if is_CNP:
            if convolutional:
                mask, context_img = image_processor(data, num_context_points, convolutional, device)
                data = data.to(device)
                batch_accuracy, batch_size = model.evaluate_accuracy(mask,context_img,target)
            else:
                x_context, y_context, x_target, y_target = image_processor(data, num_context_points, convolutional,
                                                                           device)
                batch_accuracy, batch_size = model.evaluate_accuracy(x_context,y_context,target)
        else:
            batch_accuracy, batch_size = model.evaluate_accuracy(data, target)
        sum += batch_accuracy * batch_size
        total += batch_size
    return sum/total

def test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=False,num_context_points=784, save_freq=20, is_CNP=False):

    # get the optimal epoch number
    N = 10 # window size for the smoothing the loss (moving average)
    epoch = find_optimal_epoch_number(validation_loss_dir_txt, save_freq=save_freq, window_size=N)

    # load the corresponding model
    load_dir = model_save_dir.copy()
    load_dir[5] = str(epoch)
    load_dir = "".join(load_dir)

    # load the model
    model.load_state_dict(torch.load(load_dir, map_location=device))

    # get the accuracy
    accuracy = test_model_accuracy(model, test_data, device, convolutional=convolutional, is_CNP=is_CNP)

    return accuracy

def find_optimal_epoch_number(validation_loss_dir_txt, save_freq=20, window_size = 10):

    # read in the validation loss
    l = len(validation_loss_dir_txt)
    losses =  []
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
    _, idx = min((val, idx) for (idx, val) in enumerate(moving_aves))

    # get the closest corresponding epoch number for which a checkpoint was saved
    epoch = int(round((idx + 1) / save_freq) * save_freq)
    epoch = max(epoch,save_freq) # ensure that it does not return epoch 0, at least take the first checkpoint

    return epoch


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