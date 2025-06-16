import torch.nn as nn
import torch

import torchvision
from torchvision import transforms
image_path = './'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5)),])
mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=True)
example, lable = next(iter(mnist_dataset))
print(f'Min: {example.min()} Max: {example.max()}')
print(example.shape)

def create_noise(batch_size, z_size, mode_z):
   if mode_z == 'uniform':
      input_z = torch.rand(batch_size, z_size, 1 ,1 )* 2 -1
   elif mode_z == 'normal':
      input_z = torch.randn(batch_size, z_size, 1, 1)
   return input_z

# FIX: Move fixed_z to the same device as the model


z_size = 100
image_size  = (28, 28)
n_filters = 32
image_size = (28, 28)
z_size = 100

from torch.utils.data import DataLoader
batch_size = 32
dataloader = DataLoader(mnist_dataset, batch_size, shuffle=True)
input_real, label = next(iter(dataloader))
input_real = input_real.view(batch_size, -1)
torch.manual_seed(1)
mode_z = 'uniform'
input_z = create_noise(batch_size, z_size, mode_z)

import torch
print(torch.__version__)
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = "cpu"

print(device)

import torchvision
from torchvision import transforms
image_path = './'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5)),])
mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=True)
example, lable = next(iter(mnist_dataset))
print(f'Min: {example.min()} Max: {example.max()}')
print(example.shape)

fixed_z = create_noise(batch_size, z_size, mode_z).to(device)
mnist_dl = DataLoader(mnist_dataset, batch_size = batch_size, shuffle=True, drop_last=True)
epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100


def make_genearator_network_wgan(input_size, n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4, 1, 0, bias=False),
        nn.InstanceNorm2d(n_filters*4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters*2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters, 1,4,2,1, bias=False),
        nn.Tanh()
    )
    return model

class DiscriminatorWGAN(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(0)
    
gen_model = make_genearator_network_wgan(z_size, n_filters).to(device)
disc_model = DiscriminatorWGAN(n_filters).to(device)
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)

from torch.autograd import grad as torch_grad
def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)
    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=device)
    interpolated = alpha * real_data + (1- alpha)  * generated_data
    # Calculated probability of interpolated examples
    proba_interpolated = disc_model(interpolated)
    # Calculated gradients of probabilities
    gradients = torch_grad(
        outputs = proba_interpolated, inputs=interpolated, 
        grad_outputs=torch.ones(proba_interpolated.size(), device=device),
        create_graph=True, retain_graph = True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm -1) ** 2).mean()

def d_train_wgan(x):
   disc_model.zero_grad()
   batch_size = x.size(0)
   x = x.to(device)
   d_real = disc_model(x)
   input_z = create_noise(batch_size, z_size, mode_z).to(device)
   g_output = gen_model(input_z)
   d_generated = disc_model(g_output)
   d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output)
   d_loss.backward()
   d_optimizer.step()
   return d_loss.data.item()

def g_train_wgan(x):
   gen_model.zero_grad()
   batch_size = x.size(0)
   input_z = create_noise(batch_size, z_size, mode_z).to(device)
   g_output = gen_model(input_z)
   d_generated = disc_model(g_output)
   g_loss = -d_generated.mean()
   g_loss.backward()
   g_optimizer.step()
   return g_loss.data.item()

def create_samples(g_model, input_z):
  g_output = g_model(input_z)
  images = torch.reshape(g_output, (batch_size, *image_size))
  return (images+1)/2.0

epoch_samples_wgan = []
lambda_gp = 10.0
num_epochs = 100
torch.manual_seed(1)
critic_iterations = 5
for epoch in range(1, num_epochs+1):
   gen_model.train()
   d_losses, g_losses = [], []
   for i, (x, _) in enumerate(mnist_dl):
      for _ in range(critic_iterations):
         d_loss = d_train_wgan(x)
      d_losses.append(d_loss)
      g_losses.append(g_train_wgan(x))

   print(f'Epoch {epoch:03d} | D Loss {torch.FloatTensor(d_losses).mean():.4f}')
   gen_model.eval()
   epoch_samples_wgan.append(create_samples(gen_model, fixed_z).detach().cpu().numpy())
    


import matplotlib.pyplot as plt
selected_epochs = [1, 2, 4, 10, 50, 100]
fig = plt.figure(figsize=(10, 14))
for i,e in enumerate(selected_epochs):
   for j in range(5):
      # FIX: Corrected add_subplot (not add_subplots)
      ax = fig.add_subplot(6, 5, i*5+j+1)
      ax.set_xticks([])
      ax.set_yticks([])
      if j == 0:
         ax.text(-0.06, 0.5, f'Epoch {e}', rotation=90, size = 18, color='red', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
      image = epoch_samples[e-1][j]
      ax.imshow(image, cmap='gray_r')

plt.show()