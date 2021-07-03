import torch
import random

def discriminate_same_image(model, x):
    if x.shape[0] > 1:
        x_part1, x_part2 = x.split(x.shape[0] // 2, dim=0)

        x_same = torch.cat([x_part1, x_part2], dim=-1)
        logit_same = model.discriminator(x_same)
        probs_same = model.discriminator_activation(logit_same)

        if x_part1.shape[0] != 1:

            indices = (torch.arange(0, x_part1.shape[0]) + random.randint(1, x_part1.shape[0] - 1)) % x_part1.shape[0]
            x_part2_deranged = x_part2[indices]
            x_diff = torch.cat([x_part1, x_part2_deranged], dim=-1)
            logit_diff = model.discriminator(x_diff)
            probs_diff = model.discriminator_activation(logit_diff)

            probs_same, probs_diff = torch.squeeze(probs_same, dim=-1), torch.squeeze(probs_diff, dim=-1)

            probs_same_image = torch.cat([probs_same, probs_diff], dim=-1)
        else:
            probs_same_image = torch.squeeze(probs_same, dim=-1)
    else:
        probs_same_image = torch.ones(1,device=x.device)

    return probs_same_image