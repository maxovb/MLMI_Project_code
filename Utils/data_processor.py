import numpy as np
import torch

def image_processor(data,num_context_points,convolutional=False,semantic_blocks=None,device=torch.device('cpu')):
    """ Process the image to generate the context and target points

    Args:
        data: batch of input images
        context points (int): number of context points to extract
        convolutional (bool): if the model is a ConvCNP or not
        semantic_blocks (list of strings or None): the type of way to create context, could be any subset of ["blocks", "cut", "pizza", "random"], if None: random
        device (torch.device): device to load the tensors on, i.e. CPU or GPU
    Returns:
        if convolutional == False:
            x_context (tensor): x values of the context points (batch,num_context,input_dim_x)
            y_context (tensor): y values of the context points (batch,num_context,input_dim_y)
            x_target (tensor): x values of the target points (batch,num_target,input_dim_x)
            y_target (tensor): y values of the target points (batch,num_target,input_dim_y)
        if convolution == True:
            mask (tensor): binary mask indicating the postion of context points (batch, 1, img_height, img_width)
            context_img (tensor): context img, i.e. the masked image (batch, num_channels, img_height, img_width)
       """

    # grab shapes
    batch_size, num_channels, img_height, img_width = data.shape[0], data.shape[1], data.shape[2], data.shape[3]

    # normalise pixel values
    if torch.max(data) > 1:
        data = data / 255.0

    if not convolutional:
        if not semantic_blocks:
            for i in range(batch_size):
                x, y = get_context_indices_semantic(data[i].permute(1,2,0),"random", num_context_points, convolutional=convolutional, device=device)
                if i == 0:
                    x_context = torch.zeros((batch_size,x.shape[0],x.shape[1]))
                    y_context = torch.zeros((batch_size, y.shape[0], y.shape[1]))
                x_context[i] = x
                y_context[i] = y
        if semantic_blocks:
            type_block = np.random.choice(semantic_blocks, 1, replace=False)
            if type_block == "blocks":
                percentage_active_blocks = 0.1 + 0.9 * np.random.rand()
            else:
                percentage_active_blocks = None
            if type_block == "pizza":
                percentage_active_slices = 0.1 + 0.9 * np.random.rand()
            else:
                percentage_active_slices = None
            for i in range(batch_size):
                x, y = get_context_indices_semantic(data[i].permute(1,2,0), type_block, num_context_points, convolutional=convolutional, device=device, percentage_active_blocks=percentage_active_blocks, percentage_active_slices=percentage_active_slices)
                if i == 0:
                    x_context = torch.zeros((batch_size,x.shape[0],x.shape[1]))
                    y_context = torch.zeros((batch_size, y.shape[0], y.shape[1]))
                x_context[i] = x
                y_context[i] = y

        # get the target values
        # create shell containing all indices (this is the full x data)
        image_indices = np.array([[i, j] for i in range(img_width) for j in range(img_height)])

        x_target = np.repeat(image_indices[np.newaxis, :], batch_size, axis=0)
        # normalize the x values in [0,1]
        x_target = x_target / np.array([img_height - 1, img_width - 1])

        # flatten image into vector (this is the full y data)
        y_target = torch.flatten(data, start_dim=2, end_dim=3)
        y_target = y_target.permute(0, 2, 1)

        # convert to pytorch tensors
        x_context = x_context.float().to(device)
        y_context = y_context.float().to(device)
        x_target = torch.from_numpy(x_target).type(torch.float).to(device)
        y_target = y_target.to(device)

        return x_context, y_context, x_target, y_target

    else:
        masks = torch.zeros(batch_size,1,img_width,img_height)
        image_context = torch.zeros(batch_size,num_channels,img_width,img_height)
        if not semantic_blocks:
            for i in range(batch_size):
                mask, context = get_context_indices_semantic(data[i].permute(1,2,0), "random", num_context_points, convolutional=convolutional, device=device)
                mask = mask.float().permute(2, 0, 1)
                context = context.float().permute(2, 0, 1)
                masks[i] = mask
                image_context[i] = context
        if semantic_blocks:
            type_block = np.random.choice(semantic_blocks, 1, replace=False)
            for i in range(batch_size):
                mask, context = get_context_indices_semantic(data[i].permute(1,2,0), type_block, num_context_points, convolutional=convolutional, semantic_blocks=semantic_blocks, device=device)
                mask = mask.float().permute(2, 0, 1)
                context = context.float().permute(2, 0, 1)
                masks[i] = mask
                image_context[i] = context

        return masks, image_context



    """
    if not convolutional:
        # create shell containing all indices (this is the full x data)
        image_indices = np.array([[i, j] for i in range(img_width) for j in range(img_height)])

        x_target = np.repeat(image_indices[np.newaxis, :], batch_size, axis=0)
        # normalize the x values in [0,1]
        x_target = x_target / np.array([img_height - 1, img_width - 1])

        # flatten image into vector (this is the full y data)
        y_target = torch.flatten(data, start_dim=2, end_dim=3)
        y_target = y_target.permute(0, 2, 1)

        # choose context points
        context_indices = np.zeros((batch_size, num_context_points), dtype=np.int32) # memory pre allocation for the context indices
        if not(semantic_blocks):
            for i in range(batch_size):
                context_indices[i, :] = np.random.choice(int(img_height * img_width), size=num_context_points, replace=False)
        if semantic_blocks:
            type_block = np.random.choice(semantic_blocks,1, replace=False)
            for i in range(batch_size):
                context_indices[i, :] = get_context_indices_semantic(data[i],type_block)

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
    """

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


def get_context_indices_semantic(img,type_block,num_context_points,convolutional=False, device=torch.device('cpu'), percentage_active_blocks=None, percentage_active_slices=None):
    assert type_block in ["blocks", "cut", "pizza", "random"], "type block should be one of " + ",".join(["blocks", "cut", "pizza", "random"]) + " not " + str(type_block)

    if type_block == "blocks":
        assert percentage_active_blocks, "The percentage of active blocks must be passed as an argument to use blocks as context"
    elif type_block == "pizza":
        assert percentage_active_slices, "The percentage of active pizza slices must be passed as an argument to use blocks as context"

    if type_block == "blocks":
        return context_indices_blocks(img, convolutional=convolutional, device=device, percentage_active_blocks=percentage_active_blocks)
    elif type_block == "cut":
        return context_indices_cut(img, convolutional=convolutional, device=device)
    elif type_block == "pizza":
        return context_indices_pizza(img, convolutional=convolutional, device=device, percentage_active_slices=percentage_active_slices,)
    elif type_block == "random":
        return context_indices_random(img, num_context_points, convolutional=convolutional, device=device)


def context_indices_blocks(img,blocks_per_dim=4,convolutional=False, device=torch.device('cpu'), percentage_active_blocks=1/2):
    """ Divide the image into blocks_per_dim**2 blocks and include each of them with probability 1/2 in the context

            Args:
                img (tensor): image to take the context form (img_width,img_height,num_channels)
                blocks_per_dim (int): number of blocks along each dimension, so there will be blocks_per_dim**2 blocks in total
                convolutional (bool): indicating whether the context should be return as an image and a mask
                device (torch.device): device to load the tensors on, i.e. CPU or GPU
                percentage_active_blocks (float): percentage of blocks to activate

            Returns:
                if convolutional:
                    np.array: mask (img_width,img_height)
                    np.array: masked image (img_width,img_height,num_channels)
                if not convolutional:
                    np.array: context_indices, indices of the context pixels selected (num_context_pixels,2)
                    np.array: context_values, values of the image at the context pixels selected (num_context_pixels,num_channels)
        """
    img_height,img_width = img.shape[0],img.shape[1]
    block_size_height = img_height//blocks_per_dim
    block_size_width = img_width // blocks_per_dim
    mask = torch.zeros((img_height,img_width))

    num_active_blocks = round(percentage_active_blocks * blocks_per_dim**2)
    assert num_active_blocks != 0, "The number of actives blocks must be large than 0, so the percentage of actives blocks must be larger than " + str(
        1 / (2 * blocks_per_dim**2))

    actives_blocks = np.random.choice(blocks_per_dim**2, num_active_blocks, replace=False)

    block_id = -1
    while torch.max(mask).item() == 0: # ensures that we don't return an empty array
        for i in range(blocks_per_dim):
            for j in range(blocks_per_dim):
                block_id += 1
                if block_id in actives_blocks:
                    mask[i*block_size_height:(i+1)*block_size_height,j*block_size_width:(j+1)*block_size_width] = 1
    if convolutional:
        mask = torch.unsqueeze(mask, -1)
        return mask, img*mask
    else:
        x1, x2 = (mask == 1).nonzero(as_tuple=True)
        y_ctxt = img[mask.bool()]
        return torch.stack((x1, x2), axis=1).float() / torch.Tensor([img_height - 1, img_width - 1]), y_ctxt

def context_indices_cut(img,convolutional=False, device=torch.device('cpu')):
    """ Return the context indices where we have half of the image either right/left half or top/bottom half

        Args:
            img (tensor): image to take the context form (img_width,img_height,num_channels)
            convolutional (bool): indicating whether the context should be return as an image and a mask
            device (torch.device): device to load the tensors on, i.e. CPU or GPU

        Returns:
            if convolutional:
                np.array: mask (img_width,img_height)
                np.array: masked image (img_width,img_height,num_channels)
            if not convolutional:
                np.array: context_indices, indices of the context pixels selected (num_context_pixels,2)
                np.array: context_values, values of the image at the context pixels selected (num_context_pixels,num_channels)
        """
    img_height, img_width = img.shape[0], img.shape[1]
    mask = torch.zeros((img_height, img_width))
    if np.random.rand() <= 1/2: #cut horizontally
        if np.random.rand() <= 1 / 2:
            mask[:img_height//2,:] = 1 # top half
        else:
            mask[img_height//2:, :] = 1 # bottom half
    else: # cut vertically
        if np.random.rand() <= 1 / 2:
            mask[:,:img_width//2] = 1 # left half
        else:
            mask[:,img_width//2:] = 1 # right half

    if convolutional:
        mask = torch.unsqueeze(mask, -1)
        return mask, img*mask
    else:
        x1, x2 = (mask == 1).nonzero(as_tuple=True)
        y_ctxt = img[mask.bool()]
        return torch.stack((x1, x2), axis=1).float() / torch.Tensor([img_height - 1, img_width - 1]), y_ctxt

def context_indices_pizza(img,convolutional=False, device=torch.device('cpu'),percentage_active_slices=1/2):
    """ Return the context indices where each of the pizza slice has probability 1/2 of being included in the context

    Args:
        img (tensor): image to take the context form (img_width,img_height,num_channels)
        convolutional (bool): indicating whether the context should be return as an image and a mask
        device (torch.device): device to load the tensors on, i.e. CPU or GPU
        percentage_active_slices (float): percentage of slices to activate

    Returns:
        if convolutional:
            np.array: mask (img_width,img_height)
            np.array: masked image (img_width,img_height,num_channels)
        if not convolutional:
            np.array: context_indices, indices of the context pixels selected (num_context_pixels,2)
            np.array: context_values, values of the image at the context pixels selected (num_context_pixels,num_channels)
    """

    img_height, img_width = img.shape[0], img.shape[1]
    mask = torch.ones((img_height,img_width))

    num_active_slices = round(percentage_active_slices*8)
    assert num_active_slices != 0, "The number of actives slices must be large than 0, so the percentage of actives slices must be larger than " + str(1/16)

    actives_slices = np.random.choice(8,num_active_slices,replace=False)

    slice_id = -1
    for i in range(2):
        for j in range(2):
            block = mask[i * (img_height // 2):(i + 1) * (img_height // 2), j * (img_width // 2):(j + 1) * (img_width // 2)]
            size = block.shape
            local_mask = torch.zeros(size)

            slice_id += 1
            if slice_id in actives_slices:
                remove_lower_half_diagonal = torch.eye(size[0])
                remove_lower_half_diagonal[round(img_height/4):, :] = 0
                if i == j:
                    add = torch.triu(torch.ones(size)) - remove_lower_half_diagonal
                else:
                    add = torch.flip(torch.triu(torch.ones(size)) - remove_lower_half_diagonal,[1])
                local_mask += add

            slice_id += 1
            if slice_id in actives_slices:
                remove_upper_half_diagonal = torch.eye(size[0])
                remove_upper_half_diagonal[:round(img_height/4),:] = 0
                if i == j:
                    add = torch.tril(torch.ones(size)) - remove_upper_half_diagonal
                else:
                    add = torch.flip(torch.tril(torch.ones(size)) - remove_upper_half_diagonal,[1])
                local_mask += add
            mask[i * (img_height // 2):(i + 1) * (img_height // 2), j * (img_width // 2):(j+ 1) * (img_width // 2)] *= local_mask #torch.clamp(local_mask,max=1)

    if convolutional:
        mask = torch.unsqueeze(mask, -1)
        return mask, img*mask
    else:
        x1, x2 = (mask == 1).nonzero(as_tuple=True)
        y_ctxt = img[mask.bool()]
        return torch.stack((x1, x2), axis=1).float() / torch.Tensor([img_height - 1, img_width - 1]), y_ctxt

def context_indices_random(img,num_context_points,convolutional=False, device=torch.device('cpu')):
    """ Return the context indices with by selecting random pixels

        Args:
            img (tensor): image to take the context form (img_width,img_height,num_channels)
            num_context_points (int): number of context points to keep
            convolutional (bool): indicating whether the context should be return as an image and a mask
            device (torch.device): device to load the tensors on, i.e. CPU or GPU

        Returns:
            if convolutional:
                np.array: mask (img_width,img_height)
                np.array: masked image (img_width,img_height,num_channels)
            if not convolutional:
                np.array: context_indices, indices of the context pixels selected (num_context_pixels,2)
                np.array: context_values, values of the image at the context pixels selected (num_context_pixels,num_channels)
        """
    img_height, img_width = img.shape[0], img.shape[1]

    # create the mask in ordered position
    mask = torch.zeros((img_height, img_width)).int()
    mask = torch.flatten(mask)
    mask[:num_context_points] = 1

    # shuffle it
    indices = torch.randperm(img_height * img_width)
    mask = mask[indices]
    mask = torch.reshape(mask, (img_height, img_width)).float()

    if convolutional:
        mask = torch.unsqueeze(mask,-1)
        return mask, img*mask
    else:
        x1, x2 = (mask == 1).nonzero(as_tuple=True)
        y_ctxt = img[mask.bool()]
        return torch.stack((x1, x2), axis=1).float() / torch.Tensor([img_height - 1, img_width - 1]), y_ctxt

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_height = 8
    img_width = 8
    y = np.random.random((img_height,img_width))
    x_context, y_context, x_target, y_target = image_processor(torch.from_numpy(np.expand_dims(np.expand_dims(y,axis=0),axis=0)), num_context_points=10, device=device, semantic_blocks = ["cut","blocks","pizza","random"])
    mask,context_img = image_processor(torch.from_numpy(np.expand_dims(np.expand_dims(y,axis=0),axis=0)), num_context_points=10, convolutional=True, device=device, semantic_blocks = ["cut","blocks","pizza","random"])
    print(mask)

