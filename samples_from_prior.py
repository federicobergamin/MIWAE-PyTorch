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
from utils_missing_data import create_mcar_mask, create_mar_mask, get_imputation
from torchvision import utils

# get the args
args = parse_args_MIWAE()

wandb.init(project='baseline-samples',  entity="fedbe")
wandb.config.update(args)
#

# set the seeds
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get model
weights_directory = 'paper_baseline_fully_observed/' #'paper_baseline_missing_data/'

observation_dim = 1
latent_dim = args.latent_dim

# I have to define the model and the optimizer
model = ConvVAE(input_channel=observation_dim, latent_dim=latent_dim)
# model.load_state_dict(torch.load(weights_directory + 'models/best_baseline_model_based_on_validation_log_like0_iwae_50_test_valid_with_more_samples.pt'))
model.load_state_dict(torch.load(weights_directory + 'models/final_model_eot_seed0_iwae_50_iwae_samples_and_tested_100.pt'))

model.to(device)

model.eval()

with torch.no_grad():
    samples_from_prior = model.prior.sample([64, latent_dim])
    # print(samples_from_prior.shape)

    # pass it through the decoder
    _logits, output_dist = model.decoder(samples_from_prior.to(device))

    prob = torch.sigmoid(_logits)
    print(prob.shape)
    _samples = output_dist.sample()
    print(_samples.shape)

    _sampled_images = utils.make_grid(_samples)
    _sampled_prob = utils.make_grid(prob)

    # now I can create the imshow and store them
    plt.imshow(_sampled_images[0].cpu().numpy(), cmap='gray')
    # plt.show()
    # plt.savefig(saving_directory + 'samples/' + 'samples_epoch_{}.png'.format(epoch + 1))
    wandb.log({"samples from prior": plt.imshow(_sampled_images[0].cpu().numpy(), cmap='gray')})

    plt.imshow(_sampled_prob[0].cpu().numpy(), cmap='gray')
    # plt.show()
    # plt.savefig(saving_directory + 'samples/' + 'probs_epoch_{}.png'.format(epoch + 1))
    wandb.log({"probs from prior": plt.imshow(_sampled_prob[0].cpu().numpy(), cmap='gray')})