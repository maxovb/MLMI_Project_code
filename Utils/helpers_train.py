# This was originally copied (but then largely modified) from https://github.com/brianlan/pytorch-grad-norm/blob/067e4accaa119137fca430b23c413a2bee8323b6/train.py
import numpy as np
import random
import torch
from torch import nn

class GradNorm():
    """ Class to use to apply GradNorm iterations

    Args:
        model (nn.Module): model on which to apply the GradNorm iterations
        gamma (float): hyper-parameter for the grad norm (alpha in the paper)
        ratios (list of float):  lis of objective ratios for the loss terms for the unsupervised [0] and supervsied [1] losses, if None do not use ratios
        theoretical_minimum_loss (list of float): minimum loss achievable for each loss
    Reference: modified version from
        Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
         International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(self, model, gamma, ratios=None, theoretical_minimum_loss=None):
        self.model = model
        self.gamma = gamma
        self.ratios = ratios
        self.theoretical_minimum_loss = theoretical_minimum_loss

        self.list_norms = []
        self.list_task_loss = []
        self.initial_task_loss = None

        self.list_task_weights_to_write = []

    def grad_norm_iteration(self):

        # compute the inverse training rate
        avg_task_loss = sum(self.list_task_loss)/len(self.list_task_loss)
        print("avg loss", avg_task_loss)

        if self.initial_task_loss is None: # first epoch store the loss
            self.initial_task_loss = avg_task_loss
        print("initial_task_loss",self.initial_task_loss)

        # compute the average norm 
        avg_norm = sum(self.list_norms) / len(self.list_norms)
        if 0 in avg_norm:
            print("------ FAIL: 0 in avg_norm ------")
            print("list:",self.list_norms)
            print("weights:",self.model.task_weights)

        if self.ratios:
            multiplicative_term = np.array(self.ratios)
        else:
            multiplicative_term = np.ones(avg_norm.shape)

        if self.theoretical_minimum_loss:
            min_loss = np.array(self.theoretical_minimum_loss)
            loss_ratio = (avg_task_loss - min_loss) / (self.initial_task_loss - min_loss)
            print("loss ratio", loss_ratio)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            print("inverse train rate", inverse_train_rate)
        else:
            inverse_train_rate = np.ones(avg_norm.shape)

        target_norm = np.mean(avg_norm) * (inverse_train_rate ** self.gamma)


        self.model.task_weights = torch.from_numpy(target_norm / avg_norm).to(self.model.task_weights.device)
        normalization_cst = len(self.model.task_weights) / torch.sum(self.model.task_weights, dim=0).detach()
        self.model.task_weights = self.model.task_weights * normalization_cst * multiplicative_term

        # append the new task weights 
        self.list_task_weights_to_write.append(self.model.task_weights.detach().cpu().numpy())

        # empty the list of norms and task weights
        self.list_norms = []
        self.list_task_loss = []
        
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
            norms.append(torch.norm(gygw[0]))

        norms = torch.stack(norms).detach().cpu().numpy()
        norms = torch.clip(norms,a_min=1e-2,a_max=None)

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

