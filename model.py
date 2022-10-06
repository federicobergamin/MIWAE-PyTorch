'''
Here I should implement the MIWAE model by Pierre, and other models
'''
import math

import torch
import torch
import numpy as np
from torch import nn
from torch.distributions import Normal, Binomial, MixtureSameFamily, Bernoulli, Categorical, Independent
import matplotlib.pyplot as plt
from torchvision import utils
from math import ceil

class View(nn.Module):
    """ For reshaping tensors inside Sequential objects"""
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)




# NOTE: I am not following Salimans, because the dimensionalities does not make sense
class Encoder(nn.Module):
    def __init__(self, channel_input: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.channel_input = channel_input
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            )

        self.output_layers = nn.Sequential(nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 2*self.latent_dim))

    def forward(self, x, n_samples=None):

        _out = self.feature_extractor(x)
        # print(_out.shape)
        # I forget to flatten
        _out = _out.view(-1, 4*4*32)
        _out = self.output_layers(_out)

        _mu = _out[:, 0:self.latent_dim]
        _log_var = _out[:, self.latent_dim:]

        # reparametrization trick
        dist = Normal(_mu, (0.5 * _log_var).exp())
        if n_samples is None:
            _z = dist.rsample()
        else:
            _z = dist.rsample([n_samples])

        return _mu, _log_var, _z, dist


# also this is a bit different because we cannot use stride 2 when padding is same in pytorch
class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_channel: int):
        super(BernoulliDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channel = output_channel

        # self.init_layers = nn.Sequential(nn.Linear(self.latent_dim, 300),
        #                                  nn.Softplus(),
        #                                  nn.Linear(300, 4*4*32))

        self.conv_layers = nn.Sequential(nn.Linear(latent_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 512),
                                        nn.ReLU(),
                                        View((-1, 32, 4, 4)),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        # View((-1, 28**2)),
                                    )
                                         # nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2,  padding=1))


    def forward(self, latent, n_samples=None):

        # _out = self.init_layers(latent)
        #
        # if n_samples is None:
        #     _out = _out.view(-1, 32, 4, 4)
        # else:
        #     # print(_out.shape)
        #     _out = _out.view(latent.shape[1] * n_samples, 32, 4, 4)

        # print(latent.shape)
        _logits =  self.conv_layers(latent)

        # print('_logits shape after conv layer: ', _logits.shape) #[batch_size*n_sample, ]
        if n_samples is not None:
            _logits = _logits.view(n_samples, latent.shape[1], 28, 28) # here then logits are of the shape batch_size * iwae_samples * 28 * 28

        output_dist = Bernoulli(logits=_logits)

        return _logits, output_dist


# todo: check training--> computation of the error by masking the unobserved on
class ConvVAE(nn.Module):
    def __init__(self, input_channel: int, latent_dim: int):
        super(ConvVAE, self).__init__()

        self.input_channel = input_channel
        self.latent_dim = latent_dim

        self.encoder = Encoder(self.input_channel, self.latent_dim)
        self.decoder = BernoulliDecoder(self.latent_dim, self.input_channel)

        self.prior = Normal(0,1)

    def forward(self, input, n_sample = None, mask = None):

        _mu, _log_var, _z, q_dist = self.encoder(input, n_sample)

        # I have to compute the kl

        log_pz = self.prior.log_prob(_z)

        log_qz = q_dist.log_prob(_z)

        kl = log_qz - log_pz  # shape  [n_samples, batch_size, latent_dim]

        # print('kl shape: ', kl.shape) # [n_samples, batch_size, latent_dim]

        # print(_z.shape)
        _logits, _output_dist = self.decoder(_z, n_sample) #shape batch_size * iwae_samples * 28 * 28

        # print('Logits shape: ', _logits.shape)
        # print('input shape: ', input.shape)
        # print('dist : ', _output_dist)
        if n_sample is not None:
            log_pgivenz = _output_dist.log_prob(input.squeeze(1))  # shape [batch_size, channel, 28, 28] or [batch_size * iwae_samples * 28 * 28]
        else:
            log_pgivenz = _output_dist.log_prob(input)
        # print('log_pgivenz shape: ', log_pgivenz.shape)

        if n_sample is not None:
            batch_kl = torch.sum(kl, axis=-1)#.permute(1,0) # batch_size * n_samples
            # print('batch_kl shape: ', batch_kl.shape)

            if mask is not None:
                # print('log_pgivenz shape ', log_pgivenz.shape)
                # print('mask shape ', mask.shape)
                # I have to be careful at multiplying the mask to the correct stuff
                # print(log_pgivenz.shape)
                # print(mask.squeeze(1).shape)
                masked_log_pgivenz = log_pgivenz * mask.squeeze(1)
                # print(mult.shape)
                batch_log_pgivenz = torch.sum(masked_log_pgivenz, axis=[2, 3]) # batch_size * n_samples
            else:
                batch_log_pgivenz = torch.sum(log_pgivenz, axis=[2, 3]) # batch_size * n_samples
                # print('batch_log_pgivenz shape: ', batch_log_pgivenz.shape)

            # print(torch.mean(torch.mean(batch_log_pgivenz, dim=1)))
            # print('batch_log_pgivenz shape: ', batch_log_pgivenz.shape)
            # print('batch_kl shape: ', batch_kl.shape)
            # print('torch.mean(batch_log_pgivenz - batch_kl, dim=1) shape: ', torch.mean(batch_log_pgivenz - batch_kl, dim=1).shape)
            # print('----')
            # print('batch_kl shape: ', batch_kl.shape)
            # print('batch_log_pgivenz shape: ', batch_log_pgivenz.shape)

            _bound = batch_log_pgivenz - batch_kl  # batch_size * n_sample
            # print('_bound shape: ', _bound.shape) # batch_size * n_samples
            # print('torch.mean(_bound, dim=1) shape: ', torch.mean(_bound, dim=1).shape)
            # print(torch.mean(_bound, dim=1))
            # now I have to take the log-sum-exp over the samples and the
            # print('torch.logsumexp(bound, axis=1) shape: ', torch.logsumexp(bound, axis=1).shape)
            bound = torch.logsumexp(_bound, axis=0) - math.log(n_sample) # batch_size shape
            # print('logsumexp bound shape: ', bound.shape)
            # print(bound)
            # print(torch.mean(bound))
            # print(torch.mean(torch.mean(_bound, dim=-1)))
            # print('---')
            # print('bound shape again: ', bound.shape)
            # print('bound shape after log-sum-exp: ', bound.shape)

        else:
            batch_kl = torch.sum(kl, axis=1)

            if mask is not None:
                # print('log_pgivenz shape ', log_pgivenz.shape)
                # print('mask shape ', mask.shape)
                masked_log_pgivenz = log_pgivenz * mask
                # print('mult shape ', mult.shape)
                batch_log_pgivenz = torch.sum(masked_log_pgivenz, axis=[1, 2, 3])
            else:
                batch_log_pgivenz = torch.sum(log_pgivenz, axis=[1,2,3])

            bound = batch_log_pgivenz - batch_kl # I would like the bound to be [batch_size]
            # print(bound.shape)
        if n_sample is not None:
            avg_iwae_bound = torch.mean(bound)
            # print('IWAE: ', avg_iwae_bound)
            # print(torch.mean(_bound, dim=1).shape)
            avg_vae_bound = torch.mean(torch.mean(_bound, dim=0))
            # print('VAE: ', avg_vae_bound)
            # print('-----')
        else:
            avg_vae_bound = torch.mean(bound)
            avg_iwae_bound = torch.mean(bound)

        _output_dict = {'q_mean': _mu,
                        'q_log_var': _log_var,
                        'latents': _z,
                        'q_dist': q_dist,
                        'logits': _logits,
                        'output_dist': _output_dist,
                        'kl': batch_kl,
                        'likelihood': batch_log_pgivenz,
                        'vae_bound': avg_vae_bound,
                        'iwae_bound': avg_iwae_bound
                        }

        return _output_dict


    def sample_from_prior(self, n_samples):

        _latent = self.prior.sample([n_samples, self.latent_dim])

        # now I have to pass those through the decoder
        _logits, output_dist = self.decoder(_latent, None)

        # now I have to transform these into probabilities
        probs = torch.sigmoid(_logits)
        samples = output_dist.sample()

        return probs, samples








# if __name__ == '__main__':
#
#     # I check that using the vae bound or forcing to get 1 sample give the same results and this
#     # happens, so most likely the implementation is correct
#
#     torch.manual_seed(0)
#     np.random.seed(0)
#
#     input_channel = 1
#     latent = 10
#     x = torch.ones((1, 1, 28, 28))
#
#     model = ConvVAE(1, 10)
#
#     _output_dict = model(x, None)
#
#     # print(_output_dict['vae_bound'])
#
#     _logits = _output_dict['logits']
#
#     # now I want to check few stuff
#     # I think I can transform the logits into probabilities in two ways:
#
#     prob = torch.sigmoid(_logits)
#
#     # or otherwise I was usually doing the following, which maybe it's wrong
#     _sampled_prob = _logits.exp()
#     _sampled_prob = _sampled_prob / (1 + _sampled_prob)
#
#     print(prob)
#     print('----')
#     print(_sampled_prob)
#





