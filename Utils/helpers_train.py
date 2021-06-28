# This was originally copied (but then largely modified) from https://github.com/brianlan/pytorch-grad-norm/blob/067e4accaa119137fca430b23c413a2bee8323b6/train.py
import torch
import numpy as np


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
            norms.append(torch.norm(torch.mul(self.model.task_weights[i], gygw[0])) + 1e-7)
        norms = torch.stack(norms).detach().cpu().numpy()

        if 0 in norms:
            #print("gygw",gygw)
            #print("params",list(W.parameters()))
            print("weights",self.model.task_weights)
            print("loss",task_loss)

        self.list_norms.append(norms)

        # store also the task_loss
        self.list_task_loss.append(task_loss.detach().cpu().numpy())

def grad_norm_iteration(model,task_loss,epoch,gamma,ratios=None):
    """ Apply a GradNorm iteration
    Args:
        model (nn.Module): model to apply the GradNorm iteration on
        task_loss (tensor): loss for all the tasks
        epoch (int): current epoch, to use to store the initial loss of the first epoch
        gamma (int): hyper-parameter for the grad norm (alpha in the paper)
        ratios (list of float):  lis of objective ratios for the loss terms for the unsupervised [0] and supervsied [1] losses, if None do not use ratios
    Reference:
        Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
         International Conference on Machine Learning. PMLR, 2018.
    """
    # remove the gradient on the task weights
    model.task_weights.grad.data = model.task_weights.grad.data * 0.0

    # only apply GradNorm when none of the tasks is Nan (i.e. include supervised loss)
    for loss in task_loss:
        if loss.item() == 0:
            return

    if epoch == 0:
        model.initial_task_loss = task_loss.detach().cpu().numpy()

    # get layer of shared weights
    W = model.get_last_shared_layer()

    # get the gradient norms for each of the tasks
    # G^{(i)}_w(t)
    norms = []
    for i in range(len(task_loss)):
        # get the gradient of this task loss with respect to the shared parameters
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        # compute the norm
        norms.append(torch.norm(torch.mul(model.task_weights[i], gygw[0])))
    norms = torch.stack(norms).detach().cpu().numpy()
    # print('G_w(t): {}'.format(norms))

    if epoch == 0:
        model.prev_epoch = 0
        if not(hasattr(model,"norms")):
            model.norms = [norms]
        else:
            model.norms.append(norms)
    else:
        avg_norm = sum(model.norms)/len(model.norms)
        target_norm = np.mean(avg_norm)
        if ratios:
            multiplicative_term = np.ones(len(task_loss))
            multiplicative_term[:len(model.task_weights)] *= ratios[0]
            multiplicative_term[len(model.task_weights):] *= ratios[1]
            target_norm = np.mean(avg_norm) * multiplicative_term

        model.task_weights = torch.from_numpy(target_norm/avg_norm).to(task_loss[0].device)




    """
    # compute the inverse training rate r_i(t)
    # \curl{L}_i
    if torch.cuda.is_available():
        loss_ratio = task_loss.data.cpu().numpy() / model.initial_task_loss
    else:
        loss_ratio = task_loss.data.numpy() / model.initial_task_loss
    # r_i(t)
    inverse_train_rate = loss_ratio / np.mean(loss_ratio)

    

    if ratios:
        multiplicative_term_unsup = np.ones(len(task_loss))
        multiplicative_term_unsup[:len(model.task_weights_unsup)] *= ratios[0]
        multiplicative_term_unsup[len(model.task_weights_unsup):] *= ratios[1]
        multiplicative_term_sup = np.ones(len(task_loss))
        multiplicative_term_sup[:len(model.task_weights_unsup)] *= ratios[1]
        multiplicative_term_sup[len(model.task_weights_unsup):] *= ratios[0]

        # compute the mean norm \tilde{G}_w(t)
        mean_norm_unsup = np.mean(norms.data.cpu().numpy()/multiplicative_term_unsup)
        mean_norm_sup = np.mean(norms.data.cpu().numpy()*multiplicative_term_sup)

        mean_norm = np.ones(len(task_loss))
        mean_norm[:len(model.task_weights_unsup)] *= mean_norm_unsup
        mean_norm[len(model.task_weights_unsup):] *= mean_norm_sup
    else:
        # compute the mean norm \tilde{G}_w(t)
        mean_norm = np.mean(norms.data.cpu().numpy())

    # compute the GradNorm loss
    # this term has to remain constant

    
    constant_term = torch.from_numpy((mean_norm * (inverse_train_rate ** gamma))).to(task_loss[0].device) 

    # print('Constant term: {}'.format(constant_term))
    # this is the GradNorm loss itself
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    #print('cst',constant_term)
    #print('loss',grad_norm_loss)
    # print('GradNorm loss {}'.format(grad_norm_loss))

    # compute the gradient for the weights
    model.task_weights.grad = torch.autograd.grad(grad_norm_loss, model.task_weights)[0]
    
    #print('grad',model.task_weights.grad)

    print("---norm",norms)
    print("-----cst", constant_term)
    print('--------grad',model.task_weights.grad)
    print("---------weight",model.task_weights)
    #print('weight',model.task_weights)

    """
