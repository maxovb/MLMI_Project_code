import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
os.chdir("../../")

from Utils.data_loader import load_joint_data_as_generator
from Utils.data_processor import image_processor, format_context_points_image, context_points_image_from_mask

if __name__ == "__main__":

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load the supervised set
    out = load_joint_data_as_generator(1,
                                       100,
                                       validation_split=0.1,
                                       percentage_unlabelled_set=0.25,
                                       data_version=0)
    train_data, validation_data, test_data, img_height, img_width, num_channels = out

    for i,(img,label) in enumerate(train_data):
        if i >= 5:
            break
        for semantic_blocks in ["random","cut","blocks","pizza"]:
            mask, context_img = image_processor(img,
                                                num_context_points=100,
                                                convolutional=True,
                                                semantic_blocks=[semantic_blocks],
                                                device=device)
            context_img = context_points_image_from_mask(mask, context_img)
            plt.figure()
            plt.imshow(context_img[0])
            plt.axis('off')
            plt.savefig("Results/figures/write_up/type_context/" + semantic_blocks + "_" + str(i) + ".svg")
            plt.close()




