import torch.nn as nn

def make_generator_network(input_size, n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(n_filters*4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters*2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1, bias = False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return model

class Discriminator(nn.Module):
    def  __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
            output = self.network(input)
            return output.view(-1, 1).squeeze(0)

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

z_size = 100
image_size  = (28, 28)
n_filters = 32
gen_model = make_generator_network(z_size, n_filters).to(device)
print(gen_model)

disc_model=Discriminator(n_filters).to(device)
print(disc_model)

loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0003)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)

def create_noise(batch_size, z_size, mode_z):
   if mode_z == 'uniform':
      input_z = torch.rand(batch_size, z_size, 1 ,1 )* 2 -1
   elif mode_z == 'normal':
      input_z = torch.randn(batch_size, z_size, 1, 1)
   return input_z

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

# train the generator
def g_train(x):
  gen_model.zero_grad()
  batch_size = x.size(0)
  input_z = create_noise(batch_size, z_size, mode_z).to(device)
  g_labels_real = torch.ones(batch_size, 1, device=device)
  g_output = gen_model(input_z)
  d_proba_fake = disc_model(g_output)
  g_loss = loss_fn(d_proba_fake, g_labels_real)
  g_loss.backward()
  g_optimizer.step()
  return g_loss.data.item()

def d_train(x):
   disc_model.zero_grad()
   batch_size = x.size(0)
   x = x.to(device)
   d_labels_real = torch.ones(batch_size, 1, device=device)
   d_proba_real = disc_model(x)
   d_loss_real = loss_fn(d_proba_real, d_labels_real)
   input_z = create_noise(batch_size, z_size, mode_z).to(device)
   g_output = gen_model(input_z)
   d_proba_fake = disc_model(g_output)
   d_labels_fake = torch.zeros(batch_size, 1, device=device)
   d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)
   d_loss = d_loss_real + d_loss_fake
   d_loss.backward()
   d_optimizer.step()
   return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()

# FIX: Move fixed_z to the same device as the model
fixed_z = create_noise(batch_size, z_size, mode_z).to(device)
mnist_dl = DataLoader(mnist_dataset, batch_size = batch_size, shuffle=True, drop_last=True)
epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100

def create_samples(g_model, input_z):
  g_output = g_model(input_z)
  images = torch.reshape(g_output, (batch_size, *image_size))
  return (images+1)/2.0

torch.manual_seed(1)
for epoch in range(1, num_epochs+1):
   d_losses, g_losses = [], []
   d_vals_real, d_vals_fake = [], []
   gen_model.train()
   for i, (x, _) in enumerate(mnist_dl):
      d_loss, d_proba_real, d_proba_fake = d_train(x)
      d_losses.append(d_loss)
      g_losses.append(g_train(x))

   # FIX: Corrected the print statement to show both G and D losses separately
   print(f'Epoch {epoch:03d} | Avg Losses >> G/D {torch.FloatTensor(g_losses).mean():.4f}/{torch.FloatTensor(d_losses).mean():.4f}')
   gen_model.eval()
   epoch_samples.append(create_samples(gen_model, fixed_z).detach().cpu().numpy())

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