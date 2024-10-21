import sys
import os 
import logging
import argparse
from tqdm import tqdm

from smdebug.pytorch import get_hook, modes

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
#logger.addHandler(logging.StreamHandler(sys.stdout)) # Comment this when using the sm hook

# Create profiler/debugger hook
hook = get_hook(create_if_not_exists=True)

class OneHotEncoder:
  def __init__(self, max_len = None, stop_signal = True):
        self.max_len = max_len
        self.stop_signal = stop_signal

  def encode(self, sequences):
      vocab = {
          'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
          'Q': 5, 'E': 6, 'G': 7, 'H':8, 'I': 9,
          'L': 10, 'K': 11, 'M': 12, 'F' : 13, 'P': 14,
          'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
      }

      if self.stop_signal:
          vec_lenght = 20
      else:
          vec_lenght = 21
      encoded_sequences = []
      max_len = 0
      for sequence in sequences:
          sequence = sequence.upper()
          encoded_sequence = []
          for aa in sequence:
              vec = [0 for _ in range(vec_lenght)]
              pos = vocab[aa]
              vec[pos] = 1
              encoded_sequence.append(vec)
          encoded_sequences.append(encoded_sequence)
          max_len = max(max_len, len(sequence))

      if self.max_len is not None:
          max_len = self.max_len
      max_len += 1
      
      if self.stop_signal:
          for sequence in encoded_sequences:
              while len(sequence) < max_len:
                  vec = [0 for _ in range(vec_lenght)]
                  vec[-1] = 1
                  sequence.append(vec)
      return np.array(encoded_sequences)

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
class DeepCNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(DeepCNN, self).__init__()

        # Convolutional block 1
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)

        #Conv block 2
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        # Conv block 3

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        "Define forward pass"
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        #Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        #Block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten for fc layers
        x = x.view(x.size(0), -1)
        #x = torch.reshape(x, (x.size(0), 9216))

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        output = self.fc3(x) #For regression return the direct output wo. any activation function

        return output

# Define the testing loop
def test(model, test_loader, device, steps=None):
    '''Define the testing loop'''
    
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

    # move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    
    # 0. Loop through epochs
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):

        # Set model to training mode
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        if hook:
            hook.set_mode(modes.TRAIN)

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
                target = target.unsqueeze(1)
                loss = criterion(target, preds)

                # Update validation loss/epochs
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
            logger.info("Epoch %d/%d, Validation MSE: %.3f", epoch, epochs, val_mse)
            logger.info("Epoch %d/%d, Validation R2: %.2f, Validation RMSE: %.3f, Validation MAE: %.3f", epoch, epochs, val_r2,
                        val_rmse, val_mae)

    logger.info("Finished training for %d epochs.", epochs)

def load_data(dataset_dir):
    '''Function to load sequence data from a given directory.
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

    sequence_data = np.array(dataset["sequence"])

    # If the data is for testing, target value would be missing
    if 'pMIC' in dataset.columns:
        target_data = np.array(dataset["pMIC"])
        return sequence_data, target_data
    else: 
        return sequence_data
    
def main(args):
    # Set compute device
    if not torch.cuda.is_available():
        args.device = "cpu"  #Reassign the device if cuda is not available
        print(f"CUDA not available, switching to {args.device}.")
    # Intance our model
    model = DeepCNN(input_size=51)

    # Set model configs
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss() 

    # Load datasets
    train_seqs, train_targets = load_data(args.train_dir)
    val_seqs, val_targets = load_data(args.val_dir)
    test_seqs = load_data(args.test_dir)
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
    test_results = test(model, test_loader, device=args.device, steps=1)
    logger.info("Test results: %s", test_results.tolist())

    # Save the model
    model_path = os.path.join(args.model_dir, "benchmark.pth")
    torch.save(model.cpu(), model_path)

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
