import os
import numpy as np
import torch
from Utils.data_processor import image_processor
from Utils.data_loader import load_data_unsupervised
from Utils.model_loader import load_unsupervised_model

if __name__ == "__main__":
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the model
    model_name = "UNetCNP"
    epoch_unsup = 250
    semantics = True

    CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
    num_down_blocks = CNP_model.CNN.num_down_blocks
    CNP_model = CNP_model.to(device)

    std_noise = 0 # standard deviation of the perturbation
    batch_size = 64
    validation_split = 0.1

    losses = []

    for layer_id in range(2*num_down_blocks):
        print("Layer id:", layer_id)

        CNP_model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics, device=device)
        num_down_blocks = CNP_model.CNN.num_down_blocks
        CNP_model = CNP_model.to(device)

        feature_map = {}
        feature_map["current"] = torch.ones((1,1,1,1,1))
        def hook_fn(m, i, o):
            o = o + std_noise * torch.randn(o.shape, device=device)
            return o
        if layer_id < num_down_blocks:
            CNP_model.CNN.h_down[layer_id].register_forward_hook(hook_fn)
        else:
            CNP_model.CNN.h_up[layer_id-num_down_blocks].register_forward_hook(hook_fn)

        train_data, valid_data, test_data = load_data_unsupervised(batch_size,validation_split=validation_split)

        for data,label in valid_data:
            if convolutional:
                mask, context_img = image_processor(data, num_context_points=784, convolutional=convolutional,semantic_blocks=None,
                                                    device=device)
                mean, std = CNP_model(mask,context_img)
                data = data.permute(0,2,3,1).to(device)
                assert data.shape == mean.shape, "Data and mean should have the same shape"
                loss = CNP_model.loss(mean,std,data).item()
            losses.append(loss)

        avg_loss = sum(losses)/len(losses)
        print("layer id:",layer_id,"loss:",avg_loss)
