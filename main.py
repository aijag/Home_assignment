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
class nn_decoder_module(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(nn_decoder_module, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.act = nn.Tanh()
        # self.linear2 = nn.Linear(int(hidden_size/2), output_size)

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.act(self.linear1(pred)).view(pred.data.shape[0],pred.data.shape[1], -1, 1)
        # x = self.linear2(pred)
        return pred

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
        classes = [0,1]
        r = nn_decoder_module(self.n, 2*self.n, len(classes))
        predictions = []

        optimizer = torch.optim.SGD(r.parameters(), lr=0.001, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()

        # Generate data for training:
        train_size = 2000
        train_out = np.random.binomial(1, 0.5, size=train_size)
        train_inp = np.array([train_out[i] + np.random.normal(0, self.sigma, self.n) for i in range(train_size)])

        for t in range(epochs):
            hidden = None
            inp = Variable(torch.Tensor(train_inp.reshape((train_inp.shape[0],-1, train_inp.shape[1]))), requires_grad=True)
            out = torch.Tensor(train_out.reshape((train_out.shape[0], -1, 1)), device=torch.device).float()
            pred = r(inp)
            optimizer.zero_grad()
            predictions.append(pred.data.numpy())
            loss = loss_func(pred, out)
            # if t % 20 == 0:
            #     print(t, loss.data[0])
            loss.backward()
            optimizer.step()

        # Test module
        y_t =np.array(self.y)
        output_y = r(torch.Tensor(y_t.reshape((y_t.shape[0], -1, 100), dtype=torch.long, device=torch.device), requires_grad=True))
        _, predicted = torch.max(output_y, 1)
        decode_c =[classes[predicted[j]] for j in range(y_t.shape[0])]
        print(decode_c)
        # decode_c = [r(Variable(torch.Tensor(self.y[i].reshape((self.y[i].shape[0], -1, 100))), requires_grad=True)) for i in range(self.num_of_samples)]
        # Test loss
        # print(loss_func(pred_t, Variable(torch.Tensor(test_out.reshape((test_inp.shape[0], -1, 1))))).data[0])
        return np.sum(self.samples_class != decode_c)/self.num_of_samples


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