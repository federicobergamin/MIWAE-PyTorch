'''
I have to understand if we are able to make the MNIST dataset fit in memory.
If it does not fit in memory I should find a way to be able to make the same fitting procedure
as we wanted to do
'''

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import imageio


def get_binarized_MNIST(split = 'Train', flatten = True, random_seed=0):
    '''
    Returning the binarized MNIST dataset.
    In addition to that we can return the label by using trick from
    tweet from Alemi

    :return: mnist data, mnist label
    '''

    # I start by loading the dataset
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    ims, labels = np.split(imageio.imread("https://i.imgur.com/j0SOfRW.png")[..., :3].ravel(), [-70000])
    ims = np.unpackbits(ims).reshape((-1, 28, 28))
    ims, labels = [np.split(y, [50000, 60000]) for y in (ims, labels)]

    if split.lower() == 'train':
        if flatten:
            return ims[0].reshape(-1, 784), labels[0]
        else:
            return ims[0], labels[0]

    elif split.lower() == 'valid':
        if flatten:
            return ims[1].reshape(-1, 784), labels[1]
        else:
            return ims[1], labels[1]

    else:
        if flatten:
            return ims[2].reshape(-1,784), labels[2]
        else:
            return ims[2], labels[2]



def get_bernoulli_MNIST(random_seed = 3, verbose = False):
    '''
    Function that returns the dynamic Bernoulli MNIST dataset
    '''
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    flatten_bernoulli = lambda x: x.view(-1).bernoulli()

    MNIST_dataset= MNIST('data/', train=True, transform=transforms.ToTensor(), download=False)

    data = []
    labels = []
    for i in range(len(MNIST_dataset)):
        labels.append(MNIST_dataset[i][1])
        fig = MNIST_dataset[i][0]
        # todo: by doing this we are binarizing the dataset once, and then it's stati
        ## so we are not changing the images during training --> maybe I should do this
        ## in a way that it is still dynamic
        data.append(flatten_bernoulli(fig).numpy())

    data = np.array(data)
    labels = np.array(labels)

    if verbose:
        print('Show an example in the dataset')
        plt.imshow(data[0].reshape(28, 28))
        plt.show()

        print('Dataset size')
        print(data.shape)
        print(labels.shape)

    return data, labels



# def get_dataloader_fully_observed(data, labels):
#


# if __name__ == "__main__":
#     data, labels = get_bernoulli_MNIST(random_seed=12, verbose=True)