import numpy as np
import torch

def image_processor(data,num_context_points,convolutional=False,device=torch.device('cpu')):
    """ Process the image to generate the context and target points

    Args:
        data: batch of input images
        context points (int): number of context points to extract
        convolutional (bool): if the model is a ConvCNP or not
        device (torch.device): device to load the tensors on, i.e. CPU or GPU
    Returns:
        if convolutional == False:
            x_context (tensor): x values of the context points (batch,num_context,input_dim_x)
            y_context (tensor): y values of the context points (batch,num_context,input_dim_y)
            x_target (tensor): x values of the target points (batch,num_target,input_dim_x)
            y_target (tensor): y values of the target points (batch,num_target,input_dim_y)
        if convolution == True:
            mask (tensor): binary mask indicating the postion of context points (batch, img_height, img_width, 1)
            context_img (tensor): context img, i.e. the masked image (batch, img_height, img_width, num_channels)
       """

    # grab shapes
    batch_size, num_channels, img_height, img_width = data.shape[0], data.shape[1], data.shape[2], data.shape[3]

    # normalise pixel values
    if torch.max(data) > 1:
        data = data / 255.0

    if not convolutional:
        # create shell containing all indices (this is the full x data)
        image_indices = np.array([[i, j] for i in range(img_width) for j in range(img_height)])
        x_target = np.repeat(image_indices[np.newaxis, :], batch_size, axis=0)

        # choose context points
        context_indices = np.zeros((batch_size, num_context_points), dtype=np.int32) # memory pre allocation for the context indices
        for i in range(batch_size):
            context_indices[i, :] = np.random.choice(int(img_height * img_width), size=num_context_points, replace=False)

        # normalize the x values in [0,1]
        x_target = x_target / np.array([img_height - 1, img_width - 1])

        # flatten image into vector (this is the full y data)
        y_target = torch.flatten(data,start_dim=2,end_dim=3)
        y_target = y_target.permute(0,2,1)

        # extract the context poitns form the target points
        x_context = x_target[np.arange(batch_size).reshape(-1, 1),context_indices,:]
        y_context = y_target[np.arange(batch_size).reshape(-1, 1),context_indices,:]

        return torch.from_numpy(x_context).type(torch.float).to(device), y_context.to(device), torch.from_numpy(x_target).type(torch.float).to(device), y_target.to(device)

    else:
        # move image to GPU if available
        data = data.to(device)

        # calculate the percentage of context points:
        percentage_context_points = num_context_points/(img_height*img_width)
        mask = torch.rand((batch_size, 1, img_height, img_width), device = device) < percentage_context_points
        mask = mask.type(torch.float)


        # obtain the masked image
        image_context = mask * data

        return mask, image_context

def format_context_points_image(x_context,y_context,img_height,img_width):
    """ Convert the context data to an RGB image with blue pixels for missing pixels

    Args:
        x_context (tensor): x-value of the context pixels (batch,num_context,2)
        y_context (tensor): y-value of the context pixels (batch,num_context,num_channels)
        img_height (int): number of vertical pixels in the image
        img_width (int): number of horizontal pixels in the image
    Returns:
        array: context image (batch,img_height,img_width,3)
    """
    x_context = x_context.detach().cpu().numpy()
    y_context = y_context.detach().cpu().numpy()
    x_context = np.round((x_context * [img_height-1,img_width-1])).astype(np.int32)
    y_context = (y_context * 255).astype(np.int32)

    batch_size = x_context.shape[0]

    image = np.zeros((batch_size,img_height,img_width, 3), dtype=np.int32)
    image[:, :, :, 2] = 255 # initialize non-context pixels to blue

    for i in range(batch_size):
        for x, y in zip(x_context[i], y_context[i]):
            if len(y) == 1:
                y = [y,y,y]
            image[i,x[0], x[1]] = y

    return image

def context_points_image_from_mask(mask,context_image):
    """ Output the context image with missing pixels in blue

        Args:
            mask (tensor): binary mask indicating pixel values (batch,num_context,1)
            context_image (tensor): context image with value 0 at misssing pixels
        Returns:
            array: context image (batch,img_height,img_width,3)
    """

    mask = mask.permute(0, 2, 3, 1)
    context_image = context_image.permute(0, 2, 3, 1)
    mask = mask.detach().cpu().numpy()
    inv_mask = 1-mask
    context_image = context_image.detach().cpu().numpy()

    if context_image.shape[-1] == 1:
        context_image = np.tile(context_image, (1,1,1,3))


    # set missing pixels to blue
    context_image[inv_mask.astype(bool)[:,:,:,0]] = [0,0,255]

    return context_image



