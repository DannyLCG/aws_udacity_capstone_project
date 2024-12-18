import os 
import sys
import logging
import argparse
from tqdm import tqdm

from smdebug.pytorch import get_hook, modes
from utils.encoder import OneHotEncoder

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.addHandler(logging.StreamHandler(sys.stdout))

# Create profiler/debugger hook
hook = get_hook(create_if_not_exists=True)

# Define the Custom Dataset 
class PeptideDataset(Dataset):
    '''Dataset class for defining our custom peptide dataset.
    Inherits from PyTorch Datasets so we use it as input for training/validation.
    '''
    def __init__(self, sequences, labels=None, max_len=50, stop_signal=True):
        '''Initialize the peptide dataset.
        --------------------------------------
        Params:
            sequences: array-like, sequences to create the dataset
        '''
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

# Define the testing loop
def test(model, test_loader, device, steps=None):
    '''Define the testing loop,'''
    logger.info("Testing started.")
    # Set model to evaluation mode
    model.eval()
    # ======================================================#
        # Set hook to eval mode
    # ======================================================#
    if hook:
        hook.set_mode(modes.EVAL)

    # Move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    all_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Predict only for the given number of batches, if any
            if steps is not None and batch_idx == steps:
                break

            # Move data to device
            data = data.to(device)
            # Make predictions for the test data
            preds = model(data)
            preds = preds.cpu().detach().numpy()

            # Store predictions
            all_preds.append(preds)
            
    all_preds = np.concatenate(all_preds, axis=0)

    return all_preds

#Define trainig loop
def train(model, train_loader, val_loader, optimizer, epochs, device, criterion):
    '''Define training/validation loop, return training and evaluation metrics.'''
    logger.info("Starting training.")
    # ====================================#
    # 1. Create the hook (created already)
    # ====================================#
    # ======================================================#
    # 2. Set hook to track the loss 
    # ======================================================#
    #if hook:
        #hook.register_loss(criterion)

    # move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    
    # 0. Loop through epochs
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        # Set model to training mode
        model.train()
        # ======================================================#
        # 3. Set hook to training mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.TRAIN)

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
            target = target.unsqueeze(1)
            loss = criterion(target, preds) #Careful, some other loss functions alternate input order
            # 5. Backward pass
            loss.backward()
            # 6. Update weights
            optimizer.step()

            # Update training loss/epoch
            train_loss += loss.item()
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(target.cpu().detach().numpy())

        # Compute training metrics
        train_loss = train_loss / len(train_loader) #Avg. loss per epoch
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

        # Set model to validation modee
        model.eval()
        # ======================================================#
        # 3.1 Set hook to training mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.EVAL)
        
        val_loss = 0
        val_preds, val_targets = [], []

        # 0
        with torch.no_grad():
            # 1
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device).float()
                # 3 forward pass
                preds = model(data)
                # 4 compute loss
                target = target.unsqueeze(1)
                loss = criterion(target, preds)

                # Update loss/epoch
                val_loss += loss.item()
                val_preds.extend(preds.cpu().detach().numpy())
                val_targets.extend(target.cpu().detach().numpy())

            # Compute validation metrics
            val_loss /= len(val_loader) #avg. loss per epoch
            val_mse = mean_squared_error(val_targets, val_preds)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(val_targets, val_preds)
            val_mae = mean_absolute_error(val_targets, val_preds)

            # Log validation metrics
            logger.info("Epoch %d/%d, Validation Loss: %.3f", epoch, epochs, val_loss)
            logger.info("Validation MSE: %.3f", val_mse)
            logger.info("Epoch %d/%d, Validation R2: %.3f, Validation RMSE: %.3f, Validation MAE: %.3f", epoch, epochs, val_r2, val_rmse, val_mae)

    logger.info("Finished training for %d epochs.", epochs)

def load_data(dataset_dir, test=False):
    '''Function to load dataset from a given directory.
    -------------------------------------------------
    Params:
        dataset_dir: str, The path pointong to the data in csv format.
    '''
    # Note: SageMaker stores training data under '/opt/ml/input/data/
    # Find the csv file in the provided dir
    file_path = [f for f in os.listdir(dataset_dir) if f.endswith("csv")][0]
    full_path = os.path.join(dataset_dir, file_path)
    # Load data from csv file and extract relevant fields
    dataset = pd.read_csv(full_path)

    # Check if the required columns exist
    if "sequence" not in dataset.columns or "pMIC" not in dataset.columns:
        raise ValueError("Dataset must contain 'sequence' and 'pMIC' columns.")
        
    sequence_data = np.array(dataset["sequence"])

    # To load testing data return only the sequences
    if test:
        return sequence_data
    else:
        target_data = np.array(dataset["pMIC"])
        return sequence_data, target_data

def model_fn(model_dir):
    '''Function to defin the inference/predict call. '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instance the model
    model = model = CNN_biLSTM_Model(input_size=51)
    model.to(device)
    
    # Load the trained model
    with open(os.path.join(model_dir, "ensemble_model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))

    model.eval()
    return model

def main(args):
    # Set compute device
    if not torch.cuda.is_available():
        args.device = "cpu"  #Reassign the device if cuda is not available
        print(f"CUDA not available, switching to {args.device}.")
        
    # Intance our model
    model = CNN_biLSTM_Model(input_size=51)

    # Set model configs
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss() 

    # Load datasets
    train_seqs, train_targets = load_data(args.train_dir)
    val_seqs, val_targets = load_data(args.val_dir)
    test_seqs = load_data(args.test_dir, test=True)
    # Create custom datasets
    train_dataset = PeptideDataset(train_seqs, train_targets)
    val_dataset = PeptideDataset(val_seqs, val_targets)
    test_dataset = PeptideDataset(test_seqs)
    # Instance data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    train(model, train_loader, val_loader, optimizer, args.epochs, args.device, criterion=loss_fn)

    # Test the model
    test_results = test(model, test_loader, device=args.device)
    logger.info("Test results: %s", test_results.tolist())

    # Save the model
    model_path = os.path.join(args.model_dir, "ensemble_model.pth")
    torch.save(model.state_dict(), model_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    # Container env vars
    parser.add_argument("--train_dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--val_dir", type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument("--test_dir", type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--model_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)

    main(args)
