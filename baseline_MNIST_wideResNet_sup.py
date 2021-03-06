import torch
import random
from torchsummary import summary
import os
from Networks.create_wide_resnet import create_wide_resnet
from Train.train_CNP_images import train_sup
from Utils.data_loader import load_supervised_data_as_generator
from Utils.helper_results import test_model_accuracy_with_best_checkpoint, plot_loss

if __name__ == "__main__":
    random.seed(1234)

    ##### LeNet #####
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # type of model
    pretrained = True
    model_name = "WideResNet" + ("_pretrained" if pretrained else "")
    print(model_name)

    cheat_validation= True # use a large validation set even if the trainign data is small

    # for continued supervised training
    train = True
    load = False
    save = False
    evaluate = True
    if load:
        epoch_start = 20  # which epoch to start from
    else:
        epoch_start = 0

    # training parameters
    num_training_samples = [10,20,40,60,80,100,600,1000,3000]

    for i,num_samples in enumerate(num_training_samples):

        if num_samples <= 200:
            batch_size = 64
            learning_rate = 5e-3
            epochs = 200
            save_freq = 20
        else:
            batch_size = 64
            learning_rate = 1e-3
            epochs = 200
            save_freq = 20

        # load the supervised set
        train_data, validation_data, test_data, img_height, img_width, num_channels = load_supervised_data_as_generator(batch_size, num_samples,cheat_validation=cheat_validation)

        # create the model
        model = create_wide_resnet(pretrained)
        model.to(device)

        # define the directories
        model_save_dir = ["saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/", model_name, "/", model_name, "_",model_size, "", "E", ".pth"]
        train_loss_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size + "_train.txt"
        validation_loss_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/" + model_name + "/loss/" + model_name + "_" + model_size +  "_validation.txt"
        loss_dir_plot = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + str(num_samples) + "S/"+ model_name + "/loss/" + model_name + "_" + model_size + ".svg"
        accuracies_dir_txt = "saved_models/MNIST/supervised" + ("_cheat_validation/" if cheat_validation else "/") + "accuracies/" + model_name + "_" + model_size + ".txt"

        # create directories for the checkpoints and loss files if they don't exist yet
        dir_to_create = "".join(model_save_dir[:3]) + "loss/"
        os.makedirs(dir_to_create, exist_ok=True)

        if load:
            load_dir = model_save_dir.copy()
            load_dir[-3] = str(epoch_start)
            load_dir = "".join(load_dir)

            if train:
                # check if the loss file is valid
                with open(train_loss_dir_txt, 'r') as f:
                    nbr_losses = len(f.read().split())

                assert nbr_losses == epoch_start, "The number of lines in the loss file does not correspond to the number of epochs"

            # load the model
            model.load_state_dict(torch.load(load_dir, map_location=device))
        else:
            # if train from scratch, check if a loss file already exists (it should not, so remove it if necessary)
            assert not (os.path.isfile(train_loss_dir_txt)), "The corresponding loss file already exists, please remove it to train from scratch"

        if train:
            avg_loss_per_epoch = train_sup(train_data, model, epochs, model_save_dir, train_loss_dir_txt,
                                                validation_data=validation_data,
                                                validation_loss_dir_txt=validation_loss_dir_txt, is_CNP = False,
                                                save_freq=save_freq,
                                                epoch_start=epoch_start, device=device, learning_rate=learning_rate)
            plot_loss([train_loss_dir_txt, validation_loss_dir_txt], loss_dir_plot)

        if save:
            save_dir = model_save_dir.copy()
            save_dir[-3] = str(epoch_start + epochs)
            save_dir = "".join(save_dir)
            torch.save(model.state_dict(), save_dir)

        if evaluate:
            if i == 0: # at the iteration over the different number of training samples
                # create directories for the accuracy if they don't exist yet
                dir_to_create = os.path.dirname(accuracies_dir_txt)
                os.makedirs(dir_to_create, exist_ok=True)

            num_context_points = 28 * 28
            accuracy = test_model_accuracy_with_best_checkpoint(model,model_save_dir,validation_loss_dir_txt,test_data,device,convolutional=False,num_context_points=num_context_points, save_freq=save_freq, is_CNP=False)

            # write the accuracy to the text file
            with open(accuracies_dir_txt, 'a+') as f:
                text = str(num_samples) +", " + str(accuracy) + "\n"
                f.write(text)
