import json
import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from nce import IndexLinear
import torch.optim as optim
from torch.autograd import Variable

CONFIG_FILE = "configurations.json"
# OUTPUT_DIR = "Output" + os.sep

# def plot_wer_graph(self, sigmas, WER, title_st="Decoder"):
    #     fig, ax = plt.subplots()
    #     ax.plot(sigmas, WER)
    #     ax.set(xlabel='$\sigma$', ylabel='WER',
    #            title=title_st)
    #     fig.savefig(title_st + ".png")
    #     plt.show()

def plot_wer_graph(self, sigmas, WER, title_st="Decoder"):
    trace = go.Scatter(x=sigmas, y=WER)
    data = [trace]
    py.plot(data, filename=title_st+".html")

# nn decoder class
class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, nlabel),
        )

    def forward(self, input):
        return self.main(input)

# Define the system class that holds the data and decoder functions
class System:
    def __init__(self, samples, n_bits, sigma, number_of_samples):
        self.samples_class = samples
        self.n = n_bits
        self.sigma = sigma
        self.num_of_samples = number_of_samples
        self.y = [samples[i] + np.random.normal(0, sigma, n_bits) for i in range(number_of_samples)]

    def average_decoder(self):
        decode_c = [np.average(self.y[i]) > 0.5 and 1 or 0 for i in range(self.num_of_samples)]
        return np.sum(self.samples_class != decode_c)/self.num_of_samples

    def nn_decoder(self, epochs):
        labels = [0,1]
        nlabel =len(labels)
        classifier = _classifier(nlabel)

        optimizer = optim.Adam(classifier.parameters())
        criterion = nn.MultiLabelSoftMarginLoss()

        
        epochs = 5
        for epoch in range(epochs):
            losses = []
            for i, sample in enumerate(train):
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)

                output = classifier(inputv)
                loss = criterion(output, labelsv)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.data.mean())
            print('[%d/%d] Loss: %.3f' % (epoch + 1, epochs, np.mean(losses)))


        # return np.sum(self.samples_class != decode_c)/self.num_of_samples


def decoders_estimation():
    with open(CONFIG_FILE) as config_file:
        configs = json.load(config_file)
    sigmas = np.linspace(configs["sigma_min"], configs["sigma_max"], configs["sigma_num"])
    c_samples = np.random.binomial(1, 0.5, size=configs["number_of_samples"])  # assume equal probability for 1 or 0 bit

    for n in configs["n_bits"]:
        WER_av = []
        WER_nn = []
        for sigma in sigmas:
            simulate = System(c_samples, n, sigma, configs["number_of_samples"])
            # average
            WER_av.append(simulate.average_decoder())
            # NN
            WER_nn.append(simulate.nn_decoder(epochs=200))

        plot_wer_graph(sigmas, WER_av, "Average Decoder {} - bit length".format(n))
        plot_wer_graph(sigmas, WER_nn, "NN Decoder {} - bit length".format(n))


if __name__ == "__main__":
    decoders_estimation()