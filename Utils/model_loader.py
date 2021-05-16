import torch
from CNPs.create_model import  create_model

def load_unsupervised_model(model_name, epoch, semantics = False, device = torch.device('cpu')):
    model_load_dir = ["saved_models/MNIST/", model_name + ("_semantics" if semantics else ""), "/", model_name + ("_semantics" if semantics else ""), "_", str(epoch), "E", ".pth"]
    load_dir = "".join(model_load_dir)

    # create the model
    model, convolutional = create_model(model_name)

    # load the checkpoint
    model.load_state_dict(torch.load(load_dir, map_location=device))

    return model,convolutional
