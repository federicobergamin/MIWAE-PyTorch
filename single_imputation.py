import argparse
import os
import numpy as np
import torch
from datautils import get_bernoulli_MNIST, get_binarized_MNIST
from torch.distributions import Bernoulli
from torch.utils.data import TensorDataset, DataLoader
from model import ConvVAE
import matplotlib.pyplot as plt
from math import ceil
import pickle as pkl
# import wandb
from tqdm import tqdm
from parser import parse_args_MIWAE
from utils_missing_data import create_mcar_mask, create_mar_mask, get_imputation
from torchvision import utils

missing_pattern = 'MAR'
use_wandb = True

# if use_wandb:
#     import wandb
#     wandb.init(project='baseline-missing-data-imputation',  entity="fedbe")
#     wandb.config.update(args)
# get the args
args = parse_args_MIWAE()

if use_wandb:
    import wandb
    wandb.init(project='baseline-missing-data-imputation',  entity="fedbe")
    wandb.config.update(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #
# wandb.init(project='baseline-missing-data-imputation',  entity="fedbe")
# wandb.config.update(args)
# #
weights_directory = 'paper_baseline_missing_data/'
# os.makedirs(saving_directory + 'models/', exist_ok=True)
# os.makedirs(saving_directory + 'samples/', exist_ok=True)

def get_block(matrix, row_start, row_end, col_start, col_end):
    if row_end > matrix.shape[0]:
        row_end = matrix.shape[0]
    if col_end > matrix.shape[1]:
        col_end = matrix.shape[1]

    return matrix[row_start:row_end, col_start:col_end], np.arange(row_start, row_end, 1), np.arange(col_start, col_end, 1)

# set the seeds
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# here I should get the MNIST binarized by larochelle
data, labels = get_binarized_MNIST(split='Train', flatten=False, random_seed=seed)
data_valid, label_valid = get_binarized_MNIST(split='Valid', flatten=False, random_seed=args.seed)
data_test, label_test = get_binarized_MNIST(split='Test', flatten=False, random_seed=args.seed)

# here I have to create the masks as before
if missing_pattern == 'MCAR':
    mask_train = create_mcar_mask(50000, 28*28, 0.5)
    mask_valid = create_mcar_mask(10000, 28*28, 0.5)
    mask_test = create_mcar_mask(10000, 28*28, 0.5)
elif missing_pattern == 'MAR':
    mask_train = create_mar_mask(data.reshape(-1,28*28))
    mask_valid = create_mar_mask(data_valid.reshape(-1,28*28))
    mask_test = create_mar_mask(data_test.reshape(-1,28*28))

# print('Checking shapes of mask')
# print(data.shape)
# print(mask_train.shape)
# we can reshape the masks
mask_train = mask_train.reshape(-1,28,28)
# mask_train = np.ones_like(data)
mask_valid = np.ones_like(data_valid)

missing_data_mask = np.ones_like(mask_train) - mask_train

# I start with a simple 0 imputation
zero_imputed_train = get_imputation(data, mask_train, missing_data_mask, 0)

plt.imshow(zero_imputed_train[0,:,:], cmap='gray')
plt.show()

# now I have to reshape the mask as the images
## model definition
observation_dim = 1
latent_dim = args.latent_dim

learning_rate = args.learning_rate
n_epoch = args.epochs
batch_size = args.batch_size

# in this setting I can easily create a dataloader i guess (I should create a more complicated dataloader when having the mask)
train_dataset = TensorDataset(torch.from_numpy(zero_imputed_train).float(), torch.from_numpy(labels), torch.from_numpy(mask_train))
valid_dataset = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(label_valid), torch.from_numpy(mask_valid))

# i can create the dataloaders
batch_size = 5
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)



# I have to define the model and the optimizer
model = ConvVAE(input_channel=1, latent_dim=latent_dim)
# model.load_state_dict(torch.load(weights_directory + 'models/best_baseline_model_based_on_validation_log_like0_iwae_50_1000epochs.pt'))
model.load_state_dict(torch.load(weights_directory + 'models/final_model_eot_seed0_iwae_50_mar.pt'))
model.to(device)

model.eval()

K = 10 # imputation samples


whole_dataset_imputations = []

with torch.no_grad():

    # now I have to impute the train dataset, let say I want to impute some batches
    for i, (batch_img, batch_label, batch_mask) in enumerate(train_loader):

        # for row in batch_mask.view(batch_mask.shape[0], -1):
        #     print(row)
        #
        # print('DOne')
        # print(batch_mask.shape)
        mask_repeated = batch_mask.reshape(batch_size,-1).repeat((K,1))
        # print(mask_repeated.shape)

        miss_feature_idxs1, miss_feature_idxs2 = torch.where(batch_mask.view(batch_mask.shape[0], -1) == 0)

        # batch_mask = torch.unsqueeze(batch_mask, 1)
        batch_img = torch.unsqueeze(batch_img, 1)
        batch_img = batch_img.to(device)
        batch_mask = batch_mask.to(device)
        mask_repeated = mask_repeated.to(device)

        # print(batch_mask.shape)
        # I have to encode the data
        _mu, _log_var, _z, q_dist = model.encoder(batch_img, n_samples=K)

        # _z has shape [iwae_samples, batch_size, latent_dim]

        # I have to compute p(z) and q(z|x)
        logp = model.prior.log_prob(_z)  #[iwae_samples, batch_size, latent_dim]
        logq = q_dist.log_prob(_z) #[iwae_samples, batch_size, latent_dim]

        logp = torch.sum(logp, dim=-1) #[iwae_samples, batch_size]
        logq = torch.sum(logq, dim=-1) #[iwae_samples, batch_size]

        # now I have also to decode the latent
        logits, _output_dist = model.decoder(_z, n_samples=K)

        # logits are  [iwae_samples, batch_size, first_dim, second_dim]
        # print(logits.shape)
        # now I have to compute the log p(xO|z)
        if K > 1:
            # print(batch_img.shape)
            # print(batch_img.squeeze(1).shape)
            # print(_output_dist)
            logpxz = _output_dist.log_prob(batch_img.squeeze(1))
        else:
            logpxz = _output_dist.log_prob(batch_img)


        # print(logpxz.shape)
        # now I have to multiply the logpxz with the mask
        logpxz2 = logpxz.view((K*batch_size,-1))
        observed_logpxz = logpxz * batch_mask
        logpxobsgivenz = torch.sum(logpxz2 * mask_repeated, 1).reshape([K, batch_size]) # shape [K, batch_size]
        # print(logpxobsgivenz.shape)
        # print(observed_logpxz.shape) # shape [K, batch_size, dim1, dim2]

        # now I have to compute the different weights
        log_unnormalized_importance_weights = logpxobsgivenz + logp - logq
        # print(log_unnormalized_importance_weights.shape) # shape [iwae_sample, batch_size]
        # print(log_unnormalized_importance_weights)

        imp_weights = torch.softmax(log_unnormalized_importance_weights,0, dtype=torch.float32)
        # print(imp_weights)
        # print('------')
        # now I have to get the missing logits
        logits = logits.view(logits.shape[0], logits.shape[1], -1) # shape [iwae_sample, batch_size, 784]

        probs = torch.sigmoid(logits)

        # print(imp_weights.dtype)
        # print(probs.dtype)
        # now I can compute the einsum
        imputations = torch.einsum('ki,kij->ij', imp_weights, probs) # batch_size * 784
        final_imputations = np.rint(imputations.cpu().numpy())
        # print(imputations.shape)
        whole_dataset_imputations.extend(final_imputations)


whole_dataset_imputations = np.array(whole_dataset_imputations)
print(whole_dataset_imputations.shape) # same shape as the original flatten vector

print(mask_train.flatten().shape)
missing_idx_trainset = np.where(mask_train.flatten()==0.)[0]
print(missing_idx_trainset[0:10])
true_values_missing_data = data.flatten()[missing_idx_trainset]
imputed_values = whole_dataset_imputations.flatten()[missing_idx_trainset]
print(missing_idx_trainset.shape)
print(true_values_missing_data.shape)
print(imputed_values.shape)
true_values_missing_data = torch.from_numpy(true_values_missing_data).int()
imputed_values = torch.from_numpy(imputed_values).int()

# now I can compute the accuracy
print('Accuracy')
print((true_values_missing_data == imputed_values).shape)
print(torch.sum(true_values_missing_data == imputed_values)/len(missing_idx_trainset))

print('F1 score')
from sklearn.metrics import f1_score
print(f1_score(true_values_missing_data, imputed_values))

#
#
# I can plot some imputations
n_examples = 64
zero_imputed_train = zero_imputed_train.flatten()
zero_imputed_train[missing_idx_trainset] = imputed_values
imputed_train = zero_imputed_train.reshape(50000,1,28,28)

# get some data
dataset_display = get_imputation(data, mask_train, missing_data_mask, 0.5)

dataset_display = dataset_display.reshape(-1,1,28,28)
data_with_missing_values = dataset_display[0:n_examples,:,:,:]
imputed_examples = imputed_train[0:n_examples,:,:,:]
data = data.reshape(-1,1,28,28)
true_examples = data[0:n_examples,:,:,:]

data_with_missing_values = torch.from_numpy(data_with_missing_values).float()
imputed_examples = torch.from_numpy(imputed_examples).float()
true_examples = torch.from_numpy(true_examples).float()

data_with_missing_values = utils.make_grid(data_with_missing_values)
imputed_examples = utils.make_grid(imputed_examples)
true_examples = utils.make_grid(true_examples)

plt.imshow(data_with_missing_values[0], cmap='gray')
plt.title('Images with missing data')
if use_wandb:
    wandb.log({"Images with missing data": plt.imshow(data_with_missing_values[0], cmap='gray')})
plt.close()

plt.imshow(imputed_examples[0], cmap='gray')
plt.title('Images imputed with single imputation')
if use_wandb:
    wandb.log({"Images imputed with single imputation": plt.imshow(imputed_examples[0], cmap='gray')})
# plt.show()
plt.close()

plt.imshow(true_examples[0], cmap='gray')
plt.title('True images in the training set')
if use_wandb:
    wandb.log({"True images in the training set": plt.imshow(true_examples[0], cmap='gray')})
# plt.show()
plt.close()


whole_dataset_imputations = torch.from_numpy(whole_dataset_imputations).float()
whole_dataset_imputations = whole_dataset_imputations.reshape(-1,1,28,28)
reconstruction = whole_dataset_imputations[0:n_examples,:,:,:]
reconstruction = utils.make_grid(reconstruction)

plt.imshow(reconstruction[0], cmap='gray')
plt.title('Reconstructions (Imputation of the whole images)')
if use_wandb:
    wandb.log({"Reconstructions (Imputation of the whole images)": plt.imshow(reconstruction[0], cmap='gray')})
# plt.show()
plt.close()










