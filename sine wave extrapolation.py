import torch
import torch.nn as nn
import numpy as np

import sys
import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')


# Creating a RNN class
class RNNnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers):

        super().__init__()

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, num_hidden, num_layers)

        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        hidden = torch.zeros(self.num_layers, batchsize, self.num_hidden)

        y, hidden = self.rnn(x, hidden)

        y = self.out(y)

        return y, hidden


N = 500


t = np.linspace(0, 30*np.pi, N)

# Creating a sine wave
x = np.sin(t+np.cos(t))

# Plotting the sine wave
plt.figure(figsize=(25, 3))
plt.plot(t, x, 'b-')
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# Creating a Tensor from a numpy.ndarray.
data = torch.from_numpy(x).float()

# Network parameters
input_size = 1
num_hidden = 12
num_layers = 1
seqlength = 32
batchsize = 1

# mean squared error loss function
lossfun = nn.MSELoss()

# creating an instance of the RNN class
net = RNNnet(input_size, num_hidden, num_layers)

# Using Stochastic Gradient Descent to optimize the network parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

numepochs = 20
losses = np.zeros(numepochs)

# Training the model
for epochI in range(numepochs):

    seglosses = []

    for timeI in range(N-seqlength):
        X = data[timeI:timeI+seqlength].view(seqlength, 1, 1)
        y = data[timeI+seqlength].view(1, 1)

        yHat, hidden_state = net(X)
        finalValue = yHat[-1]

        loss = lossfun(finalValue, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seglosses.append(loss.item())

    losses[epochI] = np.mean(seglosses)

    msg = f'Finished epoch {epochI+1}/{numepochs}'
    sys.stdout.write('\r'+msg)

# plotting the losses
plt.plot(losses, 's-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Testing the network

h = np.zeros((N, num_hidden))

yHat = np.zeros(N)/0

for timeI in range(N-seqlength):
    X = data[timeI:timeI+seqlength].view(seqlength, 1, 1)

    yy, hh = net(X)
    yHat[timeI+seqlength] = yy[-1]
    h[timeI+seqlength] = hh.detach()

# actual and predicted values
plt.figure(figsize=(25, 4))
plt.plot(data, label='actual data')
plt.plot(yHat, label='predicted')
plt.legend()
plt.show()


plt.plot(data-yHat, 'k^')
plt.xlabel('time')
plt.ylabel('error')
plt.show()


plt.plot(data[seqlength:], yHat[seqlength:], 'mo')
plt.xlabel('real data')
plt.ylabel('predicted data')
plt.show()


r = np.corrcoef(data[seqlength:], yHat[seqlength:])[0, 1]
print(f'correlation coefficient = {r:.2f}')


plt.figure(figsize=(16, 5))
plt.plot(h)
plt.xlabel('time')
plt.ylabel('hidden state')
plt.show()
