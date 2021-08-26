import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.utils as vutils
from PIL import Image
from ..data.two_dim import data_distrib
import os, sys
matplotlib.use('Agg')



DPI = 200

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)

def save_sequence(num_steps, x, name='', im_dir='./im', gif_dir = './gif', xlim=None, ylim=None, ipf_it=None, freq=1):
    if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)

    # PARTICLES (INIT AND FINAL DISTRIB)

    plot_paths = []
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'particle_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
            plt.plot(x[0, :, 0], x[0, :, 1], '*')
            plt.plot(x[k, :, 0], x[k, :, 1], '*')
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)
                
            #plt.axis('equal')
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths.append(filename)

    # TRAJECTORIES

    N_part = 10
    filename = name + 'trajectory.png'
    filename = os.path.join(im_dir, filename)
    plt.clf()
    plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
    plt.plot(x[0, :, 0], x[0, :, 1], '*')
    for j in range(N_part):
        xj = x[:, j, :]
        plt.plot(xj[:, 0], xj[:, 1], 'g', linewidth=2)
        plt.plot(xj[0,0], xj[0,1], 'rx')
        plt.plot(xj[-1,0], xj[-1,1], 'rx')
    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)

    make_gif(plot_paths, output_directory=gif_dir, gif_name=name)

    # REGISTRATION

    colors = np.cos(0.1 * x[0, :, 0]) * np.cos(0.1 * x[0, :, 1])

    name_gif = name + 'registration'
    plot_paths_reg = []
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'registration_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*', alpha=0)
            plt.plot(x[0, :, 0], x[0, :, 1], '*', alpha=0)
            plt.scatter(x[k, :, 0], x[k, :, 1], c=colors)
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)            
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

    # DENSITY

    name_gif = name + 'density'
    plot_paths_reg = []
    npts = 100
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'density_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            else:
                xlim = [-15, 15]
                ylim = [-15, 15]
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)                            
            plt.hist2d(x[k, :, 0], x[k, :, 1], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]], bins=npts)
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    
            



class Plotter(object):

    def __init__(self):
        pass

    def plot(self, x_tot_plot, net, i, n, forward_or_backward):
        pass

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward)


class ImPlotter(object):

    def __init__(self, im_dir = './im', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.num_plots = 100
        self.num_digits = 20
        self.plot_level = plot_level
        

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:,:self.num_plots]
            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)
            
            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)         
            
            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_first.png')
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_final.png')
                vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0,num_steps-1,self.num_plots, dtype=int) 

                for k in plot_steps:
                    # save png
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))    
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)
                    

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


class TwoDPlotter(Plotter):

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.gammas = gammas

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence(num_steps=self.num_steps, x=x_tot_plot, name=name, xlim=(-15,15),
                      ylim=(-15,15), ipf_it=ipf_it, freq=self.num_steps//min(self.num_steps,50),
                      im_dir=self.im_dir, gif_dir=self.gif_dir)


    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)
