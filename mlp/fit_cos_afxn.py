import os

from PIL import Image

import numpy as np
import random

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from IPython import display

# Define a cosine activation function
class Cosine_afxn(torch.nn.Module):
    def __init__(self):
        super(Cosine_afxn, self).__init__()
        return
    def forward(self, x):
        return torch.cos(x)

# Define a sigmoid activation function
class Sigmoid_afxn(torch.nn.Module):
    def __init__(self):
        super(Sigmoid_afxn, self).__init__()
        return
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

# Define the Absolute Cosine Similarity Loss Function
class AbsoluteCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(AbsoluteCosineSimilarityLoss, self).__init__()
        return

    def forward(self, y_true, y_pred):
        y_true_normalized = torch.nn.functional.normalize(y_true, p=2, dim=1)
        y_pred_normalized = torch.nn.functional.normalize(y_pred, p=2, dim=1)
        return torch.mean(torch.abs(1 - torch.sum(y_true_normalized * y_pred_normalized, dim=1)))
    
# Settings
x_samples = 100
x_range = [0, x_samples]
x_reshape = [-1, 1]

nn_samp_range = int(x_samples * 1.0)
# nn_start_idx = np.random.randint(0, x_samples-nn_samp_range)
nn_start_idx = 0
nn_samples = int(x_samples * 1.0)

max_plots = 20
gif_fps = 5

num_in_nodes = 1
num_out_nodes = 1

train_seed = 123
nn_lr = 1e-2
loss_th = 1e-5

# activation = nn.Sigmoid
# activation = nn.Tanh
# activation = nn.Mish
activation = Cosine_afxn
# activation = Sigmoid_afxn
# activation = nn.CosineEmbeddingLoss

loss_function = AbsoluteCosineSimilarityLoss()

afxn_type = 'cos'

# Standard function params
freq = (2 * np.pi) / (x_samples - 1)
# freq = 1
std_fxn_params = [1, 0, freq, 0]

# Generate the data from Analytical Solution
scale_ind = 1
shift_ind = 0
scale_dep = freq
shift_dep = 0
req_fxn_params = [scale_ind, shift_ind, scale_dep, shift_dep]


# Save a gif file
def save_gif_PIL(outfile, files, fps=5, loop=0):
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)


def plot_result(x, y, x_data, y_data, yh, xp=None):
   
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
   
    plt.plot(x,y, color="grey", linewidth=3, alpha=0.5, label="Exact solution")
   
    plt.plot(x,yh, color="tab:blue", linewidth=1, linestyle='dashed', alpha=1.0, label="Neural network prediction")
   
    if x_data is not None:
        plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.75, label='Training data')
       
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=20, color="tab:green", alpha=0.75,
                    label='Physics loss training locations')
   
    l = plt.legend(loc='best', frameon=False, fontsize="large")
   
    plt.setp(l.get_texts(), color="k")
    
#     plt.xlim(-0.05, 1.05)
#     plt.ylim(min(y)-0.1*abs(max(y)), max(y) + 0.1*abs(max(y)))
   
    plt.text(1.065,0.7*max(y),"Training step: %i"%(i+1),fontsize="xx-large",color="k")
   
    plt.axis("off")


# Fully Connected Network (FCN) for Training
class FCN(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT):
        
        super().__init__()
        
        # Input Layer
        self.fcn = nn.Sequential(*[nn.Linear(N_INPUT, N_OUTPUT, bias=False), activation()])
        
    def forward(self, x):
        x = self.fcn(x)
        return x


# Create a Sigmoid Function
def create_sigmoid(x, fxn_params):
    sig_eq = r"f(x) = ${\frac{%d} {1+e^{-%dx-%d}}} + %d$" % (fxn_params[0], fxn_params[2], fxn_params[3], fxn_params[1])
    lin_eq = fxn_params[2]*x + fxn_params[3]
    y = (fxn_params[0] / (1 + np.exp(-lin_eq))) + fxn_params[1]
    return y, sig_eq


# Create a Cosine Function
def create_cosine(x, fxn_params):
    cos_eq = r"f(x) = $%3.2f * cos({%3.2fx+%3.2f}) + %3.2f$" % (fxn_params[0], fxn_params[2], fxn_params[3], fxn_params[1])
    lin_eq = fxn_params[2]*x + fxn_params[3]
    y = fxn_params[0] * np.cos(lin_eq) + fxn_params[1]
    return y, cos_eq


def create_target_fxn(x, fxn_params, type='sig'):
    
    if type == 'sig':
        y, eq = create_sigmoid(x, fxn_params)
        
    if type == 'cos':
        y, eq = create_cosine(x, fxn_params)
        
    return y, eq


# Sampling Points
# Uniform Sampling
train_idxs = np.linspace(0, x_samples-1, x_samples).astype('int')

# Create the directory to store images
if not os.path.exists('epoch_plots'):
    os.mkdir('epoch_plots')


x_req = torch.linspace(x_range[0], x_range[1], x_samples).view(x_reshape[0],x_reshape[1])

y_ref, ref_eq = create_target_fxn(x_req, std_fxn_params, type=afxn_type)
y_ref = y_ref.view(x_reshape[0],x_reshape[1])

y_req, req_eq = create_target_fxn(x_req, req_fxn_params, type=afxn_type)
y_req = y_req.view(x_reshape[0],x_reshape[1])


# Slice out a small number of points
x_data = x_req[train_idxs]
y_data = y_req[train_idxs]


# Plot the original and modified functions
plt.figure()
plt.plot(x_req, y_ref, c='r', ls='dashed', label='Ref Fxn')
plt.plot(x_req, y_req, label='Exact Solution')
plt.scatter(x_data, y_data, color='tab:orange', label="Training Data")

plt.title(ref_eq, fontsize=16)
plt.legend()
# plt.show()


# Train the NN to fit the data
torch.manual_seed(train_seed)

num_epochs = 50000
loss_scale = 1

plot_freq = int(num_epochs/max_plots)


# Create the model
model = FCN(num_in_nodes, num_out_nodes)

# Assign an Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=nn_lr)


# Train the model
files_nn = []
for i in range(num_epochs):
    
    plot_img = False
    stop_training = False

    # My calculation
    # w = model.fcn[0].weight[0]
    # b = model.fcn[0].bias
    # my_yp = torch.cos(w*x_data+b)
    
    # Library calculation
    optimizer.zero_grad()
    yp_data = model(x_data)

    # Check for match
    # fwd_match = torch.all(torch.round(my_yp - yp_data, decimals=5) == 0).item()
    # if fwd_match is False:
    #     print('My calculation deviates from library')
    
    # Use MSE
    # loss_data = torch.mean((yp_data-y_data)**2)
    loss_data = loss_function(yp_data, y_data)
    print(loss_data.item())
    if loss_data < loss_th/loss_scale:
        plot_img = True
        stop_training = True
        
    if not stop_training:

        # My calculation
        # dw = -x_data * torch.sin(w*x_data+b)
        # w = w - nn_lr * torch.mean(dw)
        # db = -torch.sin(w*x_data+b)
        # b = b - nn_lr * torch.mean(db)

        # Library calculation
        loss_data.backward()
        optimizer.step()
        
    if i % plot_freq == 0:
        plot_img = True
        
    # Plot the result as the training progresses
    if plot_img:
        yp_exact = model(x_req).detach()
        
        plot_result(x_req[:,0], y_req[:,0], x_data[:,0], y_data[:,0], yp_exact)
        
        file_nn = "epoch_plots/nn_%.4i.png" % (i)
        plt.savefig(file_nn, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor='white')
        files_nn.append(file_nn)
        
        plt.close('all')
        
    if stop_training:
        break



# Plot the final fit
yp_exact = model(x_req).detach()
plot_result(x_req[:,0], y_req[:,0], x_data[:,0], y_data[:,0], yp_exact)

file_nn = "nn_sig_with_%s_%1.2f-%1.2f-%1.2f-%1.2f.png" % (activation().__class__.__name__, scale_ind, shift_ind, scale_dep, shift_dep)
plt.savefig(file_nn, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor='white')

save_gif_PIL("nn_sig_with_%s_%1.2f-%1.2f-%1.2f-%1.2f.gif" % (activation().__class__.__name__, scale_ind, shift_ind, scale_dep, shift_dep), files_nn, fps=gif_fps, loop=0)

display.Image(file_nn)

for name,param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

        