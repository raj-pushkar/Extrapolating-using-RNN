# Sine wave RNNnet

## Extrapolating the sine wave using Recurrent neural network

$$x = sin(t + cos(t))$$

![sine wave](./sine%20wave.svg)

Applied a multi-layer Elman RNN with $tanh$ non-linearity to the input sine wave

For each element in the input sequence, layer computes the following function:

$$
h_t = tanh(x_t W_{ih}^T +b_{ih}+h_{t-1} W_{hh}^T + b_{hh})
$$

where $h_t$ is the hidden state at time $t$, $x_t$​ is the input at time $t$, and $h_{(t-1)}$​ is the hidden state of the previous layer at time $t-1$ or the initial hidden state at time 0.

## Results

After training the network for 20 epochs-

![](./actual%20%26%20predicted%20vs%20t.svg)

correlation coefficient between real data and predicted data = 1.00
![](./real%20vs%20predicted.svg)
