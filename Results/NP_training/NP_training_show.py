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
    model_name = "UNetCNP"
    epoch_unsup = 400
    semantics = True
    semantic_blocks = ["cut", "blocks","pizza","random"]

    std_noise = 15  # standard deviation of the perturbation
    cumulative = True
    quantify = False

    for semantic_blocks in [None,  ["cut", "blocks","pizza","random"]]:
        for perturbation_type in ["noise", "zero", "zero-other"]:
            num_context_points = 100

            batch_size = 1
            validation_split = 0.1

            for n_restrict in [[3,6], None]:
                for model_name in ["CNP","ConvCNP","UNetCNP"]:

                    os.chdir("../")
                    model, convolutional = load_unsupervised_model(model_name, epoch_unsup, semantics=semantics,
                                                                   device=device)
                    model = model.to(device)
                    if n_restrict:
                        if type(n_restrict) == int:
                            restrict_val = n_restrict
                        else:
                            restrict_val = str(n_restrict[0]) + "-"  + str(n_restrict[1])

                    visualisation_dir = "figures/write_up/NP_training/"  + model_name + \
                                        ("_semantics_blocks" if semantic_blocks else "") + \
                                        ("_restrict" + str(restrict_val) if n_restrict else "") + ".svg"

                    train_data, valid_data, test_data = load_data_unsupervised(batch_size,
                                                                               validation_split=validation_split)
                    show_label = False
                    gen_title = model_name
                    if model_name == "CNP" or n_restrict == None:
                        show_label = True


                    os.chdir("Results/")

                    qualitative_evaluation_images(model, valid_data, num_context_points, device, visualisation_dir,
                                                  convolutional, semantic_blocks, show_label=show_label,
                                                  n_restrict=n_restrict, gen_title=gen_title)

