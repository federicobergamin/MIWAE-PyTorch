# file where I try to train a convVAE on the static MNIST dataset
import argparse
import os
import numpy as np
import torch
from datautils import get_bernoulli_MNIST, get_binarized_MNIST
from torch.utils.data import TensorDataset, DataLoader
from model import ConvVAE
import matplotlib.pyplot as plt
from math import ceil
import pickle as pkl
import wandb
from tqdm import tqdm
from parser import parse_args_MIWAE
from torchvision import utils



# get the args
args = parse_args_MIWAE()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project='baseline-fully-observed-comparison',  entity="fedbe")
wandb.config.update(args)
#
saving_directory = 'paper_baseline_fully_observed/'
# os.makedirs(saving_directory + 'models/', exist_ok=True)
# os.makedirs(saving_directory + 'samples/', exist_ok=True)

# set the seeds
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# here I should get the MNIST binarized by larochelle
data, labels = get_binarized_MNIST(split='Train', flatten=False, random_seed=seed)
data_valid, label_valid = get_binarized_MNIST(split='Valid', flatten=False, random_seed=args.seed)
data_test, label_test = get_binarized_MNIST(split='Test', flatten=False, random_seed=args.seed)

## model definition
observation_dim = 1
latent_dim = args.latent_dim

learning_rate = args.learning_rate
n_epoch =  args.epochs
batch_size = args.batch_size

# in this setting I can easily create a dataloader i guess (I should create a more complicated dataloader when having the mask)
train_dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels))
valid_dataset = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(label_valid))

# i can create the dataloaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)


# I have to define the model and the optimizer
model = ConvVAE(input_channel=1, latent_dim=latent_dim)
model = model.to(device)
print(model)
wandb.log({'model': model})
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

best_valid_log_likelihood = -10000000000
n_iwae_training = 50
n_iwae_testing = None

for epoch in tqdm(range(n_epoch)):
    tmp_kl = 0
    tmp_likelihood = 0
    tmp_vae_elbo = 0
    tmp_iwae_elbo = 0
    obs_in_epoch = 0

    for i, (batch_img, batch_label) in enumerate(train_loader):
        optim.zero_grad()

        obs_in_epoch += len(batch_img)

        batch_img = torch.unsqueeze(batch_img, 1)
        batch_img = batch_img.to(device)
        # for now I am not using IWAE bound
        _output_dict_ = model(batch_img, n_iwae_training)

        if n_iwae_training is not None:
            loss = -_output_dict_['iwae_bound']
        else:
            loss = -_output_dict_['vae_bound']

        loss.backward()
        optim.step()

        # tmp_kl += torch.sum(_output_dict_['kl']).item()
        # tmp_likelihood += torch.sum(_output_dict_['likelihood']).item()
        if n_iwae_training is not None:
            tmp_kl += torch.sum(torch.mean(_output_dict_['kl'], dim=0)).item()
            tmp_likelihood += torch.sum(torch.mean(_output_dict_['likelihood'], dim=0)).item()
        else:
            tmp_kl += torch.sum(_output_dict_['kl']).item()
            tmp_likelihood += torch.sum(_output_dict_['likelihood']).item()
        # print('2: ', loss.item() * len(obs))
        tmp_vae_elbo += _output_dict_['vae_bound'].item() * len(batch_img)
        tmp_iwae_elbo += _output_dict_['iwae_bound'].item() * len(batch_img)

    # at the end of the epoch I can print and save a log
    # print('tmp_likelihood: ', tmp_likelihood)
    # print('tmp_vae_elbo: ', tmp_vae_elbo)
    # print('tmp_iwae_elbo: ', tmp_iwae_elbo)
    print(
        "epoch {0}/{1}, train VAE ELBO: {2:.2f}, train IWAE bound: {3:.2f}, train likelihod: {4:-2f}, train KL: {5:.2f}"
            .format(epoch + 1, n_epoch, tmp_vae_elbo / obs_in_epoch, tmp_iwae_elbo / obs_in_epoch,
                    tmp_likelihood / obs_in_epoch, tmp_kl / obs_in_epoch))

    wandb.log({
        "epoch": epoch + 1, "train VAE ELBO": tmp_vae_elbo / obs_in_epoch, 'train IWAE bound': tmp_iwae_elbo / obs_in_epoch,
        "train likelihod": tmp_likelihood / obs_in_epoch, "train KL": tmp_kl / obs_in_epoch})

    # ar the end of every epoch I can also evaluate the model on the validation log-likelihood
    # I can just have a "ELBO" instead of a stricter bound for model selection

    with torch.no_grad():
        model.eval()

        valid_log_like = 0
        valid_obs = 0
        for j, (valid_batch_img, valid_batch_label) in enumerate(valid_loader):
            valid_batch_img = torch.unsqueeze(valid_batch_img, 1)
            valid_batch_img = valid_batch_img.to(device)
            valid_obs += len(valid_batch_img)

            # for now I am not using IWAE bound
            _output_dict_ = model(valid_batch_img, n_iwae_testing)

            if n_iwae_testing is not None:
                valid_elbo = _output_dict_['iwae_bound'] * len(valid_batch_img)
            else:
                valid_elbo = _output_dict_['vae_bound'] * len(valid_batch_img)

            valid_log_like += valid_elbo

        # compute the final avg elbo for the validation set
        avg_valid = valid_log_like / valid_obs
        print('Validation log p(x): ', avg_valid)
        wandb.log({'Validation log p(x)':avg_valid})
        if avg_valid > best_valid_log_likelihood:
            best_valid_log_likelihood = avg_valid
            print('BEST')
            #
            # # # save model
            torch.save(model.state_dict(),
                       saving_directory + 'models/best_baseline_model_based_on_validation_log_like' + str(args.seed) + '__trained_with_' + str(n_iwae_training) + '_iwae_samples_and_tested_' + str(n_iwae_testing) + '.pt')

        model.train()

torch.save(model.state_dict(),
                       saving_directory + 'models/final_model_eot_seed' + str(args.seed) + '_iwae_' + str(n_iwae_training)  + '_iwae_samples_and_tested_' + str(n_iwae_testing) + '.pt')
#






