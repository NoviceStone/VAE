import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm


def make_gif(imgs_dir, num_imgs, t=100):
    imgs = []
    for i in range(num_imgs):
        img_name = '{}/samples-{}.png'.format(imgs_dir, i)
        temp = Image.open(img_name)
        imgs.append(temp)
    imgs[0].save(fp=os.path.join(imgs_dir, 'samples-gif.gif'), save_all=True, append_images=imgs, duration=t, loop=0)
    print('GIF done!')


def plot_elbocurve(train_elbo, test_elbo, latent_size, save_dir):
    train_elbo = np.array(train_elbo)
    test_elbo = np.array(test_elbo)
    plt.plot(train_elbo, color='b', linestyle='-', label='train')
    plt.plot(test_elbo, color='r', linestyle='-', label='test')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Lower Bound')
    plt.title('FreyFace, $N_z$={}'.format(latent_size))
    plt.savefig(os.path.join(save_dir, 'elbocurve-{}D.png'.format(latent_size)))
    print('ELBO-curve done!')


def produce_z_values(nrows, ncolumes, scenario=1):
    if scenario == 1:
        cdf_range1 = np.linspace(1e-5, 1 - 1e-5, ncolumes)
        cdf_range2 = np.linspace(1 - 1e-5, 1e-5, nrows)
        mat_z1, mat_z2 = np.meshgrid(norm.ppf(cdf_range1), norm.ppf(cdf_range2))
        z_values = np.concatenate((mat_z1.reshape(-1, 1), mat_z2.reshape(-1, 1)), axis=1)
        return z_values
    elif scenario == 2:
        z_range1 = np.linspace(-4.0, 4.0, ncolumes)
        z_range2 = np.linspace(4.0, -4.0, nrows)
        mat_z1, mat_z2 = np.meshgrid(z_range1, z_range2)
        z_values = np.concatenate((mat_z1.reshape(-1, 1), mat_z2.reshape(-1, 1)), axis=1)
        return z_values
    else:
        raise ValueError('The argument \"scenario\" must be an integer from the set {1, 2}.')


def visualize_latentspace(z, labels, save_dir):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')
    plt.scatter(z[:, 0], z[:, 1], c=labels, marker='o', edgecolors='none', cmap=cmap)
    plt.colorbar(ticks=range(10))
    plt.grid(linestyle='-.')
    plt.xlim([-5.0, 5.0])
    plt.ylim([-5.0, 5.0])
    plt.title('The Data Distribution in 2D Latent Space')
    plt.savefig(os.path.join(save_dir, 'latent_distribution.png'))
    print('Visualize latent space, done!')
