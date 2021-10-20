import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np


def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout


def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + \
        torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout


class Langevin(torch.nn.Module):

    def __init__(self, num_steps, shape, gammas, time_sampler, device=None,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]), mean_match=True):
        super().__init__()

        self.mean_match = mean_match
        self.mean_final = mean_final
        self.var_final = var_final

        self.num_steps = num_steps  # num diffusion steps
        self.d = shape  # shape of object to diffuse
        self.gammas = gammas.float()  # schedule
        gammas_vec = torch.ones(self.num_steps, *self.d, device=device)
        for k in range(num_steps):
            gammas_vec[k] = gammas[k].float()
        self.gammas_vec = gammas_vec

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()
        self.time_sampler = time_sampler

    def record_init_langevin(self, init_samples):
        mean_final = self.mean_final
        var_final = self.var_final

        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        gammas = self.gammas.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time

        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        store_steps = self.steps
        num_iter = self.num_steps
        steps_expanded = time

        for k in range(num_iter):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, mean_final, var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma)*z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx

            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)  # / (2 * gamma)

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, t_batch=None, ipf_it=0, sample=False):
        mean_final = self.mean_final
        var_final = self.var_final

        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        gammas = self.gammas.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time

        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        store_steps = self.steps
        steps_expanded = steps
        num_iter = self.num_steps

        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = net(x, steps[:, k, :])

                if sample & (k == num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z

                t_new = net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = x + net(x, steps[:, k, :])

                if sample & (k == num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, steps[:, k, :])

                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def forward(self, net, init_samples, t_batch, ipf_it):
        return self.record_langevin_seq(net, init_samples, t_batch, ipf_it)
