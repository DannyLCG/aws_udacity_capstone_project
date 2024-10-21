import torch
from torch import nn
import torch.nn.functional as F

# Create the custom PT model
class CNN_biLSTM_Model(nn.Module):
    def __init__(self, input_size, conv_filters=64, kernel_size=3, n_units=64, dropout_rate=0.5):
        super(CNN_biLSTM_Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=kernel_size, stride=1)

        # Bidirectional LSTM layers
        base_log = int((torch.log2(torch.tensor(n_units)) - 1).item())
        s_units = 2**base_log

        self.bilstm1 = nn.LSTM(input_size=conv_filters, hidden_size=n_units, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=2 * n_units, hidden_size=s_units, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(2 * s_units, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x):
        "Define forward pass"
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2) #input_tensor, kernel_size

        x = x.permute(0, 2, 1) #Permute to (batch, seq_lenght, features) for LSTMs

        x, _ = self.bilstm1(x) #LSTMs already have internal activation functions (sigmoid and tanh)
        x, _ = self.bilstm2(x) #and return 2 values: the output and hidden states
        # Take the last time step
        x = x[:, -1, :]

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = self.fc3(x) #For regression return the direct output wo. any activation function

        return output