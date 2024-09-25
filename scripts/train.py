import sys
import math
import time
import logging
import argparse
from tqdm import tqdm

from utils.encoder import OneHotEncoder

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set the logger
logger = logging.getLogger(name=__name__)
logger.setLevel(level="DEBUG")
logger.addHandler(logging.StreamHandler(sys.stdout))

# Define the Custom Dataset 
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels=None, max_len=50, stop_signal=True):
        self.sequences = sequences
        self.labels = labels
        self.encoder = OneHotEncoder(max_len=max_len, stop_signal=stop_signal)
        self.encoded_sequences = self.encoder.encode(self.sequences)
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.encoded_sequences[index]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        if self.labels is not None:
            label = self.labels[index]
            if label is None:
                raise ValueError(f"Label not found for sequence with index: {index}")
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return sequence_tensor, label_tensor
        else:
            return sequence_tensor

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

def test(model, test_loader, device):
    '''Define the testing loop,'''
    logger.info("Testing started.")

    # Move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    # Set model to evaluation mode
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data, _ in test_loader:
            # Move data to device
            data = data.to(device)
            # Make predictions for the test data
            preds = model(data)
            preds = preds.cpu().detach().numpy()

            # Store predictions
            all_preds.append(preds)
            
    all_preds = np.concatenate(all_preds, axis=0)

    return all_preds
            
def train(model, train_loader, val_loader, optimizer, epochs, device, criterion):
    '''Define training/validation loop, return training and evaluation metrics.'''
    logger.info("Starting training.")

    # move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    # Set model in training mode
    model.train()

    # 0. Loop through epochs
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        train_loss = 0
        train_preds, train_targets = [], []

        # 1. Loop through data
        for data, target in train_loader:
            # Move data to device
            data, target = data.to(device), target.to(device).float()
            # 2. Zero gradients
            optimizer.zero_grad()
            # 3. Perform forward pass
            preds = model(data)
            #preds = preds.unsqueeze(1) # reshape to (batch_size, 1)
            # 4. Compute loss
            loss =+ criterion(target, preds) #Careful, some other loss functions alternate input order
            # 5. Backward pass
            loss.backward()
            # 6. Update weights
            optimizer.step()

            # Update training loss/epoch
            train_loss += loss.item()
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(target.cpu().detach().numpy())

        # Compute training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_mse = mean_squared_error(train_targets, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Log training metrics
        logger.info("Epoch %d/%d, Training Loss: %.3f", epoch, epochs, train_loss)
        logger.info("Epoch %d/%d, Training MSE: %.3f", epoch, epochs, train_mse)
        logger.info("Epoch %d/%d, Training R2: %.2f, Training RMSE: %.3f, Training MAE: %.3f", epoch, epochs, train_r2, train_rmse, train_mae)

        # Perform validation
        logger.info("Validation started.")

        # Set model to validation mode
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            # 1
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device).float()
                # 3. Forward pass
                preds = model(data)
                #preds = preds.unsqueeze(1) # reshape to (batch_size, 1)
                # 4 Compute loss
                loss = criterion(target, preds)

                # Update validation loss/epochs
                val_loss += loss.item()
                val_preds.extend(preds.cpu().detach().numpy())
                val_targets.extend(target.cpu().detach().numpy())

            # Compute validation metrics
            val_loss /= len(val_loader.dataset)
            val_mse = mean_squared_error(val_targets, val_preds)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(val_targets, val_preds)
            val_mae = mean_absolute_error(val_targets, val_preds)

            # Log validation metrics
            logger.info("Epoch %d/%d, Validation Loss: %.3f", epoch, epochs, val_loss)
            logger.info("Epoch %d/%d, Validation MSE: %.3f", epoch, epochs, val_mse)
            logger.info("Epoch %d/%d, Validation R2: %.2f, Validation RMSE: %.3f, Validation MAE: %.3f", epoch, epochs, val_r2,
                        val_rmse, val_mae)

    logger.info("Finished training for %ds epochs.", epochs)
