# This was originally copied (but then largely modified) from https://github.com/brianlan/pytorch-grad-norm/blob/067e4accaa119137fca430b23c413a2bee8323b6/train.py
import numpy as np
import random
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

class GradNorm():
    """ Class to use to apply GradNorm iterations

    Args:
        model (nn.Module): model on which to apply the GradNorm iterations
        gamma (float): hyper-parameter for the grad norm (alpha in the paper)
        ratios (list of float):  lis of objective ratios for the loss terms for the unsupervised [0] and supervsied [1] losses, if None do not use ratios
    Reference: modified version from
        Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
         International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(self, model, gamma, ratios=None):
        self.model = model
        self.gamma = gamma
        self.ratios = ratios

        self.list_norms = []
        self.list_task_loss = []
        self.initial_task_loss = None

        self.list_task_weights_to_write = []

    def grad_norm_iteration(self):

        # compute the inverse training rate
        avg_task_loss = sum(self.list_task_loss)/len(self.list_task_loss)
        if self.initial_task_loss is None: # first epoch store the loss
            self.initial_task_loss = avg_task_loss
        loss_ratio = avg_task_loss / self.initial_task_loss
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the average norm 
        avg_norm = sum(self.list_norms) / len(self.list_norms)
        if 0 in avg_norm:
            print("------ FAIL: 0 in avg_norm ------")
            print("list:",self.list_norms)
            print("weights:",self.model.task_weights)

        if self.ratios:
            multiplicative_term = np.array(self.ratios)
            target_norm = np.mean(avg_norm) * multiplicative_term * (inverse_train_rate ** self.gamma)
        else:
            target_norm = np.mean(avg_norm) * (inverse_train_rate ** self.gamma)


        self.model.task_weights = torch.from_numpy(target_norm / avg_norm).to(self.model.task_weights.device)
        normalization_cst = len(self.model.task_weights) / torch.sum(self.model.task_weights, dim=0).detach()
        self.model.task_weights = self.model.task_weights * normalization_cst

        # append the new task weights 
        self.list_task_weights_to_write.append(self.model.task_weights.detach().cpu().numpy())

        # empty the list of norms
        self.list_norms = []
        
        print('target_norm',target_norm)
        print('weights',self.model.task_weights)
        print('avg',avg_norm)
        

    def store_norm(self, task_loss):

        for loss in task_loss:
            if loss.item() == 0:
                return

        # get layer of shared weights
        W = self.model.get_last_shared_layer()

        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(gygw[0]) + 1e-7)
        norms = torch.stack(norms).detach().cpu().numpy()

        if 0 in norms:
            print("weights",self.model.task_weights)
            print("loss",task_loss)

        self.list_norms.append(norms)

        # store also the task_loss
        self.list_task_loss.append(task_loss.detach().cpu().numpy())

    def write_to_file(self,weights_dir_txt):
        with open(weights_dir_txt,"a+") as f:
            for task_weights in self.list_task_weights_to_write:
                txt_list = [str(weight) for weight in task_weights]
                txt = ", ".join(txt_list) + " \n"
                f.write(txt)
        self.list_task_weights_to_write = []


def consistency_loss(output_logit, num_sets_of_context=1):

    batch_size = output_logit.shape[0]

    # obtain the probability distribution
    probs = nn.Softmax(dim=-1)(output_logit)

    assert num_sets_of_context == 2, "Consistency loss does not handle other number of context sets than 2 at the moment"

    # get the original batch size
    single_set_batch_size = batch_size / num_sets_of_context
    assert single_set_batch_size == int(single_set_batch_size), "The tensor batch size should be a multiple of the number of sets of context (when using consistency regularization), but got batch size: " + str(mean.shape[0]) + " and num of context sets: " + str(num_sets_of_context)
    single_set_batch_size = int(single_set_batch_size)

    # split between the two sets of context sets
    probs_set1, probs_set2 = torch.split(probs, single_set_batch_size, dim=0)
    loss = js_divergence(probs_set1,probs_set2)

    if batch_size > 2:

        assert batch_size % 2 == 0, "The batch size should be divisible by two, repeat every image twice with two context sets"

        indices = torch.ones(batch_size//2,device=probs.device)
        for i in range(batch_size//2):
            while True:
                r = random.randint(0,batch_size//2-1)
                if r != i: # check that we don't compare two same images
                    break
            indices[i] = r

        probs_compare = probs_set2[indices.type(torch.int64)]
        loss += - js_divergence(probs_set1, probs_compare)

    return loss


def js_divergence(probs_set1, probs_set2):
    """Jenson-Shannon divergence between the two probabilties distributions
    """
    # compute the Jensen Shannon divergence
    m = probs_set1 + probs_set2
    loss = 0.0
    dist1 = Categorical(probs_set1)
    dist2 = Categorical(probs_set2)
    distm = Categorical(m)
    loss += kl_divergence(dist1,distm)
    loss += kl_divergence(dist2,distm)
    div = 0.5 * torch.mean(loss)

    return div

