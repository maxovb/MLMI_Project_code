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

    def __init__(self, model, gamma, ratios=None, theoretical_minimum_loss=None, clip_value=None):
        self.model = model
        self.gamma = gamma
        self.clip_value = clip_value

        self.ratios = ratios
        self.theoretical_minimum_loss = theoretical_minimum_loss

        self.list_norms = []
        self.list_task_loss = []
        self.initial_task_loss = None

        self.list_task_weights_to_write = []
        self.list_mean_norms_to_write = []
        self.list_std_norms_to_write = []

    def scale_only_grad_norm_iteration(self):
        if self.ratios:
            multiplicative_term = torch.from_numpy(np.array(self.ratios)).to(self.model.task_weights.device)
            self.model.task_weights = multiplicative_term

        # append the new task weights 
        self.list_task_weights_to_write.append(self.model.task_weights.detach().cpu().numpy())

        # empty the lists
        self.list_norms = []
        self.list_task_loss = []

    def grad_norm_iteration(self):

        # compute the inverse training rate
        avg_task_loss = sum(self.list_task_loss)/len(self.list_task_loss)
        print("avg loss", avg_task_loss)

        if self.initial_task_loss is None: # first epoch store the loss
            self.initial_task_loss = avg_task_loss
        print("initial_task_loss",self.initial_task_loss)

        # compute the average norm
        array_norms = np.array(self.list_norms)

        if self.clip_value != None:
            array_norms = np.clip(array_norms,a_min=clip_value,a_max=None)

        avg_norm = np.mean(array_norms,dim=0)
        std_norm = np.std(array_norms,dim=0)

        if 0 in avg_norm:
            print("------ FAIL: 0 in avg_norm ------")
            print("list:",self.list_norms)
            print("weights:",self.model.task_weights)

        if self.ratios:
            multiplicative_term = torch.from_numpy(np.array(self.ratios)).to(self.model.task_weights.device)
        else:
            multiplicative_term = torch.ones(avg_norm.shape)

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
        self.list_mean_norms_to_write.append(avg_norm.tolist())
        self.list_std_norms_to_write.append(std_norm.tolist())

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

        if 0 in norms:
            print("weights",self.model.task_weights)
            print("loss",task_loss)

        self.list_norms.append(norms)

        # store also the task_loss
        self.list_task_loss.append(task_loss.detach().cpu().numpy())

    def write_epoch_data_to_file(self,gradnorm_dir_txt):
        weights_dir_txt = gradnorm_dir_txt + "task_weights.txt"
        self.write_to_file(self, weights_dir_txt, self.list_task_weights_to_write)
        self.list_task_weights_to_write = []

        mean_dir_txt = gradnorm_dir_txt + "mean_norm.txt"
        self.write_to_file(self, mean_dir_txt, self.list_mean_norms_to_write)
        self.list_mean_norms_to_write = []

        std_dir_txt = gradnorm_dir_txt + "std_norm.txt"
        self.write_to_file(self, std_dir_txt, self.list_std_norms_to_write)
        self.list_std_norms_to_write = []

    def write_to_file(self,dir_txt,list_values):
        with open(dir_txt,"a+") as f:
            for x in list_values:
                if type(x) == list:
                    txt_list = [str(val) for val in x]
                else:
                    txt_list = [str(x)]
                txt = ", ".join(txt_list) + " \n"
                f.write(txt)

