import torch
import numpy as np
from tqdm import tqdm
from Utils.data_processor import image_processor

def train_CNP(train_data,model,epochs, model_save_dir, loss_dir_txt, min_context_points=2,max_percentage_context = 0.33, convolutional=False, report_freq = 100, learning_rate=1e-3, weight_decay=1e-5, save_freq = 10, epoch_start = 0, device=torch.device('cpu')):
    img_height, img_width = train_data.dataset[0][0].shape[1], train_data.dataset[0][0].shape[2]

    # pre-allocate memory to store the losses
    avg_loss_per_epoch = []
    loss_to_write = []

    # define the optimizer
    opt = torch.optim.Adam(model.parameters(),
                           learning_rate,
                           weight_decay=weight_decay)

    for i in range(epochs):
        print("Epoch:", i+epoch_start+1)
        model.train()
        losses = []
        iterator = tqdm(train_data)
        for batch_idx, (data, target) in enumerate(iterator):
            num_context_points = np.random.randint(min_context_points,int(img_height * img_width * max_percentage_context))
            if convolutional:
                mask, context_img = image_processor(data,num_context_points,convolutional,device)
                data = data.to(device)
                loss = model.train_step(mask,context_img,data, opt)
            else:
                x_context, y_context, x_target, y_target = image_processor(data,num_context_points,convolutional,device)
                loss = model.train_step(x_context,y_context,x_target, y_target, opt)
            # store the loss
            losses.append(loss)

            if batch_idx == 0 or (batch_idx + 1) % report_freq == 0: # report the loss
                avg_loss = np.array(losses).mean()
                txt = report_loss("Training", avg_loss, batch_idx)
                iterator.set_description(txt)
                iterator.refresh()  # to show immediately the update

        # compute the average loss over the epoch and store all losses
        epoch_avg_loss = np.array(losses).mean()
        avg_loss_per_epoch.append(epoch_avg_loss)
        loss_to_write.append(epoch_avg_loss)

        # Print the average epoch loss
        print("Average loss:", epoch_avg_loss)

        # save the checkpoint
        if (i+1) % save_freq == 0:
            # update the file name
            save_dir = model_save_dir.copy()
            save_dir[5] = str(epoch_start + i+1)
            save_dir = "".join(save_dir)

            # save the model
            torch.save(model.state_dict(), save_dir)

            # write the average epoch loss to the txt file
            with open(loss_dir_txt, 'a+') as f:
                for loss_item in loss_to_write:
                    f.write('%s\n' % loss_item)
            loss_to_write = []

    # write the final losses
    with open(loss_dir_txt, 'a+') as f:
        for loss_item in loss_to_write:
            f.write('%s\n' % loss_item)

    return avg_loss_per_epoch


def report_loss(name,avg_loss,step):
    txt = name + " loss, step " + str(step) + ": " + str(avg_loss)
    return txt


def write_loss(model_save_dir,val):
    # directory
    loss_dir = model_save_dir.copy()
    loss_dir[1] = "loss_"
    loss_dir[5] = str(epochs)

    # text file with the loss
    loss_dir_txt = loss_dir.copy()
    loss_dir_txt[6] = ".txt"
    loss_dir_txt = "".join(loss_dir_txt)