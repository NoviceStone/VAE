import os
import math
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from models import VAE
from datasets import FreyFaceDataset
from utils import produce_z_values, visualize_latentspace


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" ======================================= PART 1: EXPERIMENTS ON MNIST DATASET ========================================== """

# Build the data input pipeline
batch_size = 100
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Specify the directory stored the trained model parameters
paras_dir = 'trained_parameters'
results_dir = 'results/MNIST'

# Build the model and recover the trained model parameters
model_2D = VAE(input_size=784, hidden_size=500, latent_size=2).to(device)
model_2D.load_state_dict(torch.load(os.path.join(paras_dir, 'mnist_zdim2.pkl')))

# ========================= Experiment 1: Visualization of Learned MNIST Manifold ========================= #

z_values = produce_z_values(nrows=20, ncolumes=20)
z_values = torch.from_numpy(z_values).float()
with torch.no_grad():
    generated_imgs = model_2D.decode(z_values).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, 'MNIST-manifold.png'), nrow=20)

# ============== Experiment 2: Show Data(with Labels) Distribution in Learned 2D Latent Space ============= #

# step 1: set the number of datapoints in this experiment
num_datapoints = 5000
# step 2: fetch 5000 (image, label) pairs from test-dataloader
for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
    if batch_idx < math.ceil(num_datapoints / batch_size):
        if batch_idx == 0:
            imgs = batch_x
            labs = batch_y
        else:
            imgs = torch.cat((imgs, batch_x))
            labs = torch.cat((labs, batch_y))
    else:
        break
# step 3: get 5000 (latent, label) pairs by encoding images to latents
with torch.no_grad():
    imgs = imgs.to(device).view(-1, 784)
    latents_mean, latents_logvar = model_2D.encode(imgs)
    latents = model_2D.reparameterize(latents_mean, latents_logvar)
    latents = latents.numpy()
    labs = labs.numpy()
# step 4: show all lantents with corresponding labels in 2D latent space
visualize_latentspace(latents, labs, results_dir)

# =================== Experiment 3: generation and reconstruction with different z_dim =================== #

# recover trained models with different z_dim
model_5D = VAE(input_size=784, hidden_size=500, latent_size=5).to(device)
model_5D.load_state_dict(torch.load(os.path.join(paras_dir, 'mnist_zdim5.pkl')))
model_10D = VAE(input_size=784, hidden_size=500, latent_size=10).to(device)
model_10D.load_state_dict(torch.load(os.path.join(paras_dir, 'mnist_zdim10.pkl')))
model_20D = VAE(input_size=784, hidden_size=500, latent_size=20).to(device)
model_20D.load_state_dict(torch.load(os.path.join(paras_dir, 'mnist_zdim20.pkl')))

with torch.no_grad():    
    # Generation
    noise2 = torch.randn(100, 2).to(device)
    generated_imgs = model_2D.decode(noise2).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-2D.png'), nrow=10)
    noise5 = torch.randn(100, 5).to(device)
    generated_imgs = model_5D.decode(noise5).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-5D.png'), nrow=10)
    noise10 = torch.randn(100, 10).to(device)
    generated_imgs = model_10D.decode(noise10).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-10D.png'), nrow=10)
    noise20 = torch.randn(100, 20).to(device)
    generated_imgs = model_20D.decode(noise20).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-20D.png'), nrow=10)
    # Reconstruction
    for batch_idx, (batch_x, _) in enumerate(test_loader):
        true_imgs = batch_x.view(-1, 1, 28, 28)
        save_image(true_imgs, os.path.join(results_dir, 'origin_imgs.png'), nrow=10)
        break
    x = true_imgs.to(device).view(-1, 784)
    reconst_x = model_2D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 28)
    reconst_loss1 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-2D.png'), nrow=10)
    reconst_x = model_5D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 28)
    reconst_loss2 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-5D.png'), nrow=10)
    reconst_x = model_10D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 28)
    reconst_loss3 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-10D.png'), nrow=10)
    reconst_x = model_20D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 28)
    reconst_loss4 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-20D.png'), nrow=10)


""" ====================================== PART 2: EXPERIMENTS ON FREYFACE DATASET ========================================= """

# Build the data input pipeline
batch_size = 48
dataset = FreyFaceDataset(root='./data/FreyFace', transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# Specify the directory stored the trained model parameters
paras_dir = 'trained_parameters'
results_dir = 'results/FreyFace'

# Build the model and recover the trained model parameters
model_2D = VAE(input_size=560, hidden_size=200, latent_size=2, data_type='real').to(device)
model_2D.load_state_dict(torch.load(os.path.join(paras_dir, 'freyface_zdim2.pkl')))

# ======================== Experiment 1: Visualization of Learned FreyFace Manifold ======================== #

z_values = produce_z_values(nrows=10, ncolumes=14)
z_values = torch.from_numpy(z_values).float()
with torch.no_grad():
    generated_imgs = model_2D.decode(z_values)[0].view(-1, 1, 28, 20)
    save_image(generated_imgs, os.path.join(results_dir, 'FreyFace-manifold.png'), nrow=14)

# ==================== Experiment 2: generation and reconstruction with different z_dim ==================== #

# recover trained models with different z_dim
model_5D = VAE(input_size=560, hidden_size=200, latent_size=5, data_type='real').to(device)
model_5D.load_state_dict(torch.load(os.path.join(paras_dir, 'freyface_zdim5.pkl')))
model_10D = VAE(input_size=560, hidden_size=200, latent_size=10, data_type='real').to(device)
model_10D.load_state_dict(torch.load(os.path.join(paras_dir, 'freyface_zdim10.pkl')))
model_20D = VAE(input_size=560, hidden_size=200, latent_size=20, data_type='real').to(device)
model_20D.load_state_dict(torch.load(os.path.join(paras_dir, 'freyface_zdim20.pkl')))

with torch.no_grad():
    # Generation
    noise2 = torch.randn(48, 2).to(device)
    generated_imgs = model_2D.decode(noise2)[0].view(-1, 1, 28, 20)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-2D.png'), nrow=8)
    noise5 = torch.randn(48, 5).to(device)
    generated_imgs = model_5D.decode(noise5)[0].view(-1, 1, 28, 20)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-5D.png'), nrow=8)
    noise10 = torch.randn(48, 10).to(device)
    generated_imgs = model_10D.decode(noise10)[0].view(-1, 1, 28, 20)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-10D.png'), nrow=8)
    noise20 = torch.randn(48, 20).to(device)
    generated_imgs = model_20D.decode(noise20)[0].view(-1, 1, 28, 20)
    save_image(generated_imgs, os.path.join(results_dir, 'gene_imgs-20D.png'), nrow=8)
    # Reconstruction
    for batch_idx, batch_x in enumerate(data_loader):
        true_imgs = batch_x.view(-1, 1, 28, 20)
        save_image(true_imgs, os.path.join(results_dir, 'origin_imgs.png'), nrow=8)
        break
    x = true_imgs.to(device).view(-1, 560)
    reconst_x, _ = model_2D(inputs)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 20)
    reconst_loss1 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-2D.png'), nrow=8)
    reconst_x, _ = model_5D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 20)
    reconst_loss2 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-5D.png'), nrow=8)
    reconst_x, _ = model_10D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 20)
    reconst_loss3 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-10D.png'), nrow=8)
    reconst_x, _ = model_20D(x)[-1]
    reconst_imgs = reconst_x.view(-1, 1, 28, 20)
    reconst_loss4 = torch.sum((x - reconst_x).pow(2)) / len(x)
    save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-20D.png'), nrow=8)
