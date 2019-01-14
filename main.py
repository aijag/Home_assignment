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
import torch.optim as optim

import math
from torch.autograd import Variable
import torch.nn.functional as F

CONFIG_FILE = "configurations.json"

# def plot_wer_graph(self, sigmas, WER, title_st="Decoder"):
    #     fig, ax = plt.subplots()
    #     ax.plot(sigmas, WER)
    #     ax.set(xlabel='$\sigma$', ylabel='WER',
    #            title=title_st)
    #     fig.savefig(title_st + ".png")
    #     plt.show()

def plot_wer_graph(sigmas, WER, title_st="Decoder"):
    trace = go.Scatter(x=sigmas, y=WER, mode='lines+markers', name ='Monte Carlo Simulation')
    data = [trace]
    # Edit the layout
    layout = dict(title=title_st, xaxis=dict(title='Variance'),
                  yaxis=dict(title='WER'),)
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=title_st+".html")

def tensor_round(tensor):
    return torch.round(tensor)
class autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(True),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(True),
            nn.Linear(hidden_dim3, latent_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim3),
            nn.ReLU(True),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.ReLU(True),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

    def nn_decoder(self, epochs, device, lr, weight_decay):
        # Model
        model = autoencoder(self.n, int(0.5*self.n), int(0.25*self.n), int(0.1*self.n), 1).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            # ============Generate data for training=========
            train_size = 1000
            np.random.seed(0)
            train_out = np.random.binomial(1, 0.5, size=train_size)
            train_out2 = np.array([train_out[i] + np.zeros(self.n) for i in range(train_size)])
            train_inp = np.array([train_out[i] + np.random.normal(0, self.sigma, self.n) for i in range(train_size)])
            # Convert training data to Tensor:
            inp = Variable(torch.Tensor(train_inp.reshape((train_inp.shape[0], -1, train_inp.shape[1]))),
                           requires_grad=True).to(device)
            out = Variable(torch.Tensor(train_out.reshape((train_out.shape[0], -1, 1)))).to(device)
            out2 = Variable(torch.Tensor(train_out2.reshape((train_out2.shape[0], -1, train_out2.shape[1])))).to(device)
            # ===================forward=====================
            output = model(inp)
            loss = criterion(output, out2)
            MSE_loss = nn.MSELoss()(output, out2)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(loss.item())

        # Test module
        y_t =np.array(self.y)
        output_y = model(Variable(torch.Tensor(y_t.reshape((y_t.shape[0], -1, y_t.shape[1])))).to(device))
        decode_nn = output_y.cpu().detach().numpy().reshape(self.num_of_samples,self.n)
        decode_nn = [np.average(decode_nn[i]) > 0.5 and 1 or 0 for i in range(self.num_of_samples)]
        return np.sum(self.samples_class != decode_nn)/self.num_of_samples


def decoders_estimation():
    with open(CONFIG_FILE) as config_file:
        configs = json.load(config_file)
    sigmas = np.linspace(configs["sigma_min"], configs["sigma_max"], configs["sigma_num"])
    np.random.seed(0)
    c_samples = np.random.binomial(1, 0.5, size=configs["number_of_samples"])  # assume equal probability for 1 or 0 bit
    use_cuda = configs["cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for n in configs["n_bits"]:
        WER_av = []
        WER_nn = []
        for sigma in sigmas:
            print ("Variance: {}".format(sigma))
            simulate = System(c_samples, n, sigma, configs["number_of_samples"])
            # average
            WER_av.append(simulate.average_decoder())
            # NN
            WER_nn.append(simulate.nn_decoder(epochs=configs["epochs"], device=device,lr= configs["lr"], weight_decay=configs["weight_decay"]))

        plot_wer_graph(sigmas, WER_av, "Average Decoder {} - bit length".format(n))
        plot_wer_graph(sigmas, WER_nn, "NN Decoder {} - bit length".format(n))


if __name__ == "__main__":
    decoders_estimation()