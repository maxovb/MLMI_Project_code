# This was originally copied (but then largely modified) from https://github.com/brianlan/pytorch-grad-norm/blob/067e4accaa119137fca430b23c413a2bee8323b6/train.py
import matplotlib.pyplot as plt
import numpy as np
import os
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
        clip_value (float): clip the avg norm value with this value (to avoid having one norm being very small and taking all the weight)
    Reference: modified version from
        Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
         International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(self, model, gamma, ratios=None, theoretical_minimum_loss=None, clip_value=None, losses_name=None, initial_task_loss=None, regression_loss=True):
        self.model = model
        self.gamma = gamma
        self.clip_value = clip_value
        self.losses_name = losses_name
        self.initial_task_loss = initial_task_loss
        self.regression_loss = regression_loss


        self.ratios = ratios
        self.theoretical_minimum_loss = theoretical_minimum_loss

        self.list_norms = []
        self.list_task_loss = []

        self.list_task_weights_to_write = []
        self.list_mean_norms_to_write = []
        self.list_std_norms_to_write = []

        self.trainable_loss = {}

    def scale_only_grad_norm_iteration(self):
        if self.ratios:
            multiplicative_term = torch.from_numpy(np.array(self.ratios)).to(self.model.task_weights.device)
            self.model.task_weights = multiplicative_term

        # compute the average and standard deviation of the norms
        array_norms = np.array(self.list_norms)

        avg_norm = np.mean(array_norms, axis=0)
        std_norm = np.std(array_norms, axis=0)

        # append the new task weights 
        self.list_task_weights_to_write.append(self.model.task_weights.detach().cpu().numpy())
        self.list_mean_norms_to_write.append(avg_norm.tolist())
        self.list_std_norms_to_write.append(std_norm.tolist())

        # empty the lists
        self.list_norms = []
        self.list_task_loss = []

    def grad_norm_iteration(self):

        # compute the inverse training rate
        avg_task_loss = sum(self.list_task_loss)/len(self.list_task_loss)
        
        if self.initial_task_loss is None: # first epoch store the loss
            self.initial_task_loss = avg_task_loss

        # compute the average norm
        array_norms = np.array(self.list_norms)

        avg_norm = np.mean(array_norms,axis=0)
        std_norm = np.std(array_norms,axis=0)

        if self.clip_value != None:
            avg_norm = np.clip(avg_norm, a_min=self.clip_value, a_max=None)

        if self.ratios:
            multiplicative_term = torch.from_numpy(np.array(self.ratios)).to(self.model.task_weights.device)
        else:
            multiplicative_term = torch.ones(avg_norm.shape)

        if self.theoretical_minimum_loss:
            min_loss = np.array(self.theoretical_minimum_loss)
            loss_ratio = (avg_task_loss - min_loss) / (self.initial_task_loss - min_loss)
            if self.regression_loss:
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            else:
                inverse_train_rate = loss_ratio / np.mean(loss_ratio[1:])
        else:
            inverse_train_rate = np.ones(avg_norm.shape)

        if self.regression_loss:
            target_norm = np.mean(avg_norm) * (inverse_train_rate ** self.gamma)
        else:
            target_norm = np.mean(avg_norm[1:]) * (inverse_train_rate ** self.gamma)

        if self.regression_loss:
            self.model.task_weights = torch.from_numpy(target_norm / avg_norm).to(self.model.task_weights.device)
        else:
            self.model.task_weights[1:] = torch.from_numpy(target_norm[1:] / avg_norm[1:]).to(self.model.task_weights.device)
            self.model.task_weights[0] = 0

        if 0 in multiplicative_term:
            normalization_cst = torch.sum((multiplicative_term != 0).float()) / torch.sum(self.model.task_weights * (multiplicative_term != 0).float(), dim=0).detach()
        else:
            normalization_cst = len(self.model.task_weights) / torch.sum(self.model.task_weights, dim=0).detach()
        self.model.task_weights = self.model.task_weights * normalization_cst * multiplicative_term

        # append the new task weights 
        self.list_task_weights_to_write.append(self.model.task_weights.detach().cpu().numpy())
        self.list_mean_norms_to_write.append(avg_norm.tolist())
        self.list_std_norms_to_write.append(std_norm.tolist())

        # empty the list of norms and task weights
        self.list_norms = []
        self.list_task_loss = []

    def store_norm(self, task_loss):

        for loss in task_loss[1:]:
            if loss.item() == 0 or torch.isnan(loss):
                return

        # get layer of shared weights
        W = self.model.get_last_shared_layer()

        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            if task_loss[i].requires_grad:
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(gygw[0]))
                self.trainable_loss[i] = True
            else:
                norms.append(torch.zeros(1,device=task_loss[i].device)[0])
                self.trainable_loss[i] = False

        norms = torch.stack(norms).detach().cpu().numpy()

        self.list_norms.append(norms)

        # store also the task_loss
        self.list_task_loss.append(task_loss.detach().cpu().numpy())

    def write_epoch_data_to_file(self,gradnorm_dir_txt):
        # create directories for the accuracy if they don't exist yet
        dir_to_create = os.path.dirname(gradnorm_dir_txt)
        os.makedirs(dir_to_create, exist_ok=True)

        weights_dir_txt = gradnorm_dir_txt + "task_weights.txt"
        self.write_to_file(weights_dir_txt, self.list_task_weights_to_write)
        self.list_task_weights_to_write = []

        mean_dir_txt = gradnorm_dir_txt + "mean_norm.txt"
        self.write_to_file(mean_dir_txt, self.list_mean_norms_to_write)
        self.list_mean_norms_to_write = []

        std_dir_txt = gradnorm_dir_txt + "std_norm.txt"
        self.write_to_file(std_dir_txt, self.list_std_norms_to_write)
        self.list_std_norms_to_write = []

    def write_to_file(self,dir_txt,list_values):
        with open(dir_txt,"a+") as f:
            for x in list_values:
                if isinstance(x,(list,np.ndarray)):
                    txt_list = [str(val) for val in x]
                else:
                    txt_list = [str(x)]
                txt = ", ".join(txt_list) + " \n"
                f.write(txt)

    def read_from_file(self,file_dir_txt):
        all_values = []
        with open(file_dir_txt, "r") as f:
            for i, line in enumerate(f.readlines()):
                line_values = line.split("\n")[0].split(", ")
                line_values = [[float(x)] for x in line_values]
                if i == 0:
                    all_values = line_values
                else:
                    [all_values[j].extend(line_values[j]) for j in range(len(line_values))]
        return all_values

    def plot_weights(self,gradnorm_dir_txt, losses_name=None):

        weights_dir_txt = gradnorm_dir_txt + "task_weights.txt"
        weights_dir_plot = gradnorm_dir_txt + "task_weights.svg"

        weights = self.read_from_file(weights_dir_txt)
        n = len(weights)
        l = len(weights[0])
        weights = np.array(weights)

        plt.figure()
        plt.semilogy(np.stack([np.arange(l)] *  n).T,weights.T)
        if losses_name != None:
            plt.legend(labels=losses_name,fontsize="x-large")
        plt.ylabel("GradNorm weight",fontsize=15)
        plt.xlabel("Epoch", fontsize=15)

        plt.savefig(weights_dir_plot)

    def plot_mean_and_std_norms(self,gradnorm_dir_txt, losses_name=None):

        mean_dir_txt = gradnorm_dir_txt + "mean_norm.txt"
        std_dir_txt = gradnorm_dir_txt + "std_norm.txt"
        dir_plot = gradnorm_dir_txt + "mean_std_norm.svg"

        means = self.read_from_file(mean_dir_txt)
        stds = self.read_from_file(std_dir_txt)
        l = len(means[0])
        n = len(means)
        assert l == len(stds[0]), "The number of values for the means of the norms should be the same as the stds"
        means = np.array(means)
        stds = np.array(stds)
        colors = ["blue","orange","green","red"]
        plt.figure()
        for j in range(n):
            if losses_name != None:
                plt.semilogy(np.arange(l),means[j], label=losses_name[j],color=colors[j])
            else:
                plt.semilogy(np.arange(l), means[j],color=colors[j])
            #plt.fill_between(np.arange(l),means[j] - 1.96 * stds[j], means[j] + 1.96 * stds[j], alpha=0.25,color=colors[j])
        if losses_name != None:
            plt.legend(fontsize="x-large")
        plt.ylabel("Gradient norm",fontsize=15)
        plt.xlabel("Epoch", fontsize=15)

        plt.savefig(dir_plot)

    def plot_all(self,gradnorm_dir_txt):
        self.plot_weights(gradnorm_dir_txt,self.losses_name)
        self.plot_mean_and_std_norms(gradnorm_dir_txt,self.losses_name)

if __name__ == "__main__":
    grad_norm_iterator = GradNorm(model=None,gamma=None)
    grad_norm_dir_txt = "../saved_models/MNIST/joint_semantics_cons_GN_1.5_ET/0.25P_0V/100S/UNetCNP_GMM/grad_norm/"
    losses_name = ["Regression loss","Consistency loss","Extra task classification loss","Classification loss"]
    grad_norm_iterator.plot_weights(grad_norm_dir_txt,losses_name)
    grad_norm_iterator.plot_mean_and_std_norms(grad_norm_dir_txt, losses_name)




