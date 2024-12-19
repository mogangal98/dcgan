import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import datetime as dt

import matplotlib.animation as animation
from IPython.display import HTML

seed = 98 # manually determined seed to see same results each time
random.seed(seed)
torch.manual_seed(seed)

dataroot = "C:/dataset"  #dataset directory
batch_size = 16  #pre determining the batch size for training
image_size = 256  # image size will be 64x64. this will require a size transformer since photos in the dataset is usually 640x480
nc = 3 # number of channels. 3 -> color images
nz = 100 # size of generator input vector
ngf = 200 #size of feature maps in generator
ndf = 50 #size of feature maps in discriminator
num_epochs = 10000 # number of training epochs
lr_g = 0.0001 # learning rate, if this is smaller, the network learns better but it takes a longer time
lr_d = 0.0001 # lr = 0.0001 
beta1 = 0.5 # beta param for Adam optimizer (In adam optimizer, learing rate changes during training)
beta2 = 0.75
ngpu = 1 # number of gpus

#%% Intiliazing dataset
dataset = dset.ImageFolder(root=dataroot,   #dataset directory
                           transform = transforms.Compose([ # images needs to be transformed for compatiblility with pytorch
                               transforms.Resize(image_size), # 64x64
                               transforms.CenterCrop(image_size), # crops the image, starting from center of the image
                               transforms.ToTensor(), # transforms the image to a tensor (a multidimensional matrix used in pytorch library)
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  #normalization converts the iamge to a greyscaled image. This is done to increase the performance of CNN of gen. and dis. networks
                               ]))
# creating dataloader(pytorch)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
# Dataloader is used to combine dataset and sampler.

# Deciding the device
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# In this case, Nvidia GPU based cuda will be used, but cpu can be used as well.


#%% weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# weights are randomly initialized from normal distribution with mean=0, standart_deviation= 0.02

#%% Generator
# Generator will create RGB images with the same size of training images (3x64x64)
# This will be done with a series of 2D convolutional layers with ReLU activation functions
class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu # numbe of gpu, =1
        self.main = nn.Sequential(
            
        nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 32),
        nn.LeakyReLU(0.2, inplace=True), 
        
        nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, (2,2), 1, bias=False), 
        nn.BatchNorm2d(ngf * 16),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, (2,2), 1, bias=False), 
        nn.BatchNorm2d(ngf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, (2,2), 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, (2,2), 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d( ngf * 2, ngf, 4, (2,2), 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.ConvTranspose2d( ngf, nc, 4, (2,2), 1, bias=False), # nc x 128x128
        nn.Tanh()  # orjinali tanh
        )
    def forward(self, input):
        return self.main(input)
        
    
# Creating the generator
netG = Generator(ngpu).to(device)
# Randomly initializing weigths with previously written weights_init function
netG.apply(weights_init)
#Summary of the model is printed:
# print(netG)


#%% Discriminator
# Discriminator is a binary classification network. (0 for fake images and 1 for real images)
# Discriminator takes 3x64x64 input image and processes it through a series of 2d convolutional layers
# and final layer is processed through a sigmoid function to get a final probability.
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, stride=(2,2), padding = 1 , bias=False),  #stride = 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, stride=(2,2), padding = 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=(2,2), padding = 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=(2,2), padding = 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=(2,2), padding = 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, ndf * 32, 4, stride=(2,2), padding = 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 32, 1, 4, stride=1,padding = 0, bias=False), # stride = 1
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)
    
# Create the Discriminator
netD = Discriminator(ngpu).to(device)
# Randomly initializing weigths with previously written weights_init function
netD.apply(weights_init)
#Summary of the model is printed:
# print(netD)


#%% Loss function and optimizers
# For this network, binary cross entropy (BCELoss) is used.

criterion = nn.BCELoss()

# Batch of latent vectors used to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Images will be labeld during training
real_label = np.random.uniform(low=0.8,high=1.1) # real = 1, fake = 0 ---------------------------------------------------------------------------------------------
fake_label = np.random.uniform(low=0.0,high=0.2)

# Setup Adam optimizers for both Generator and Discriminator
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
# optimizerD = optim.SGD(netD.parameters(), lr = lr_d)
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
# Adam optimizers changes the learning rate during training, for better and faster results.
# Beta hyperparameter affects how much the adam optimizer "remember" its previous movements
# Lower beta value increases epochs(number of training sets) and slows down the training

#%% Training
img_list = []
G_losses = []
D_losses = []
iters = 0

start_time = dt.datetime.now() # start time

load_checkpoint = False # True if previous checkpoints exists and they will be used
g_path = "weights/g_check.pth.tar"  # save directory for checkpoints
d_path = "weights/n_check.pth.tar"

prev_epoch = 0

for epoch in range(prev_epoch,num_epochs): # if there are no prior checkpoints, prev_epoch = 0
    errD = torch.FloatTensor([5])
    for i, data in enumerate(dataloader, 0): # For each batch in the dataloader

        # Update Discriminator network, real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # labeling
        output = netD(real_cpu).view(-1) # forward pass of real batch
        errD_real = criterion(output, label) # Calculating loss
        errD_real.backward() # Calculating gradients
        D_x = output.mean().item() # Calculating D(x)

        # Update Discriminator , fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device) # generating noise vectors
        fake = netG(noise) # generating fake images using generator
        label.fill_(fake_label) # labeling fake images
        output = netD(fake.detach()).view(-1) # passing fake images through discriminator.
        errD_fake = criterion(output, label) # calculating discrimanator loss on the fake batch
        errD_fake.backward() # calculating gradients for fake batch
        D_G_z1 = output.mean().item() # D(x)
        errD = errD_real + errD_fake # Total error is sum of error in real and fake batches
        optimizerD.step() # Update Discriminator
        

        torch.cuda.empty_cache()
        #Update Generator network
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1) # another forward pass with fake batch through now updated Discriminator
        errG = criterion(output, label) # calculate generator loss based on this
        errG.backward() # gradients
        D_G_z2 = output.mean().item() # G(z)      
        optimizerG.step() # Update Generator

        # Output training stats       
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    
    
#%% PLOT final result
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()   
    
#capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())    
    
# side by side
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
    

finish_time = dt.datetime.now() - start_time
print("Training finished at ",dt.datetime.now(),"\nTotal time:",finish_time)

#%% save all photos
zdeneme = np.transpose(img_list[-1],(1,2,0))
for i in range(1,8):
    for j in range(1,8):
        a = zdeneme[((i-1)*130):(i*130),((j-1)*130):(j*130),:]
        plt.axis("off")
        plt.imshow(a)
        isim = "fotolar_"+str(i)+"_"+str(j)+"png"
        plt.savefig(isim,transparent=True)

