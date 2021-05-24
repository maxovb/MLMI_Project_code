import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from Utils.data_processor import image_processor
from Utils.data_loader import load_data_unsupervised
from Utils.model_loader import load_unsupervised_model
from Utils.helper_results import qualitative_evaluation_images

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the model
    model_name = "ConvCNP"
    epoch_unsup = 400
    semantics = False
    num_context_points = 784
    semantic_blocks = None

    std_noise = 15  # standard deviation of the perturbation
    #perturbation_type = "zero-other"  # one of ["noise", "zero", "zero-other"]
    for perturbation_type in ["noise", "zero", "zero-other"]:

        model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
        if model_name == "CNP":
            num_layers_to_peturb = 1
        elif model_name == "ConvCNP":
            num_layers_to_peturb = model.CNN.num_residual_blocks
        elif model_name in ["UNetCNP","UNetCNP_restrained"]:
            num_layers_to_peturb = model.CNN.num_down_blocks + 1

        batch_size = 1
        validation_split = 0.1

        losses = []

        for layer_id in range(0,num_layers_to_peturb):
            print("Layer id:", layer_id)

            model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
            model = model.to(device)

            visualisation_dir = "saved_models/MNIST/" + model_name + ("_semantics" if semantics else "") \
                                + "/perturbation/" + model_name + "_L" + str(layer_id) + "_" + perturbation_type + \
                                (str(std_noise) if perturbation_type == "noise" else "") + \
                                ("_semantics_blocks.svg" if semantic_blocks else ".svg")

            loss_dir = "saved_models/MNIST/" + model_name + ("_semantics" if semantics else "") + "/perturbation/" +\
                    model_name + "_" + perturbation_type + (str(std_noise) if perturbation_type == "noise" else "") + \
                    ("_semantics_blocks.txt" if semantic_blocks else ".txt")

            if layer_id == 0:
                dir_to_create = os.path.dirname(visualisation_dir)
                os.makedirs(dir_to_create,exist_ok=True)

            def hook_fn(m, i, o):
                #print(torch.mean(torch.abs(o),dim=[0,1,2,3]))
                if perturbation_type == "noise":
                    o = o + std_noise * torch.randn(o.shape, device=device)
                elif perturbation_type in ["zero", "zero-other"]:
                    o = o * 0
                return o

            # register hook to perform the perturbation
            if perturbation_type in ["noise", "zero"]:
                if model_name == "CNP":
                    model.encoder.register_forward_hook(hook_fn)
                elif model_name in ["ConvCNP"]:
                    model.CNN.h[layer_id].register_forward_hook(hook_fn)
                elif model_name in ["UNetCNP", "UNetCNP_restrained"]:
                    model.CNN.connections[layer_id].register_forward_hook(hook_fn)
            elif perturbation_type in ["zero-other"]:
                for layer_to_perturb in range(num_layers_to_peturb):
                    if layer_to_perturb != layer_id:
                        if model_name == "CNP":
                            model.encoder.register_forward_hook(hook_fn)
                        elif model_name in ["ConvCNP"]:
                            model.CNN.h[layer_to_perturb].register_forward_hook(hook_fn)
                        elif model_name in ["UNetCNP", "UNetCNP_restrained"]:
                            model.CNN.connections[layer_to_perturb].register_forward_hook(hook_fn)

            train_data, valid_data, test_data = load_data_unsupervised(batch_size,validation_split=validation_split)

            qualitative_evaluation_images(model,valid_data,num_context_points,device,visualisation_dir,convolutional,semantic_blocks)

            losses = []
            iterator = tqdm(valid_data)
            for i,(data,label) in enumerate(iterator):
                if convolutional:
                    mask, context_img = image_processor(data, num_context_points=784, convolutional=convolutional,
                                                        semantic_blocks=None, device=device)
                    mean, std = model(mask,context_img)
                    data = data.permute(0,2,3,1).to(device)
                    assert data.shape == mean.shape, "Data and mean should have the same shape"
                    loss = model.loss(mean,std,data).item()
                else:
                    x_context, y_context, x_target, y_target = image_processor(data, num_context_points=784,
                                                                            convolutional=convolutional,
                                                                            semantic_blocks=None, device=device)
                    mean, std = model(x_context, y_context, x_target)
                    loss = model.loss(mean, std, y_target).item()
                losses.append(loss)
            avg_loss = np.array(losses).mean()

            if layer_id == 0:
                with open(loss_dir,'w') as f:
                    f.write('%s\n' % avg_loss)
            else:
                with open(loss_dir,'a') as f:
                    f.write('%s\n' % avg_loss)
