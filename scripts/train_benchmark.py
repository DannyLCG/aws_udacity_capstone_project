import sys
import os
import logging
import argparse
from tqdm import tqdm

import smdebug.pytorch as smd
from smdebug.pytorch import get_hook, modes
from utils.encoder import OneHotEncoder
from models.cnn_model import DeepCNN

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.addHandler(logging.StreamHandler(sys.stdout)) # Comment this when using the sm hook

# Create profiler/debugger hook
#hook = get_hook(create_if_not_exists=True)
hook = smd.Hook.create_from_json_file() # To record the losses

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
    # ====================================#
    # 1. Create the hook (created already)
    # ====================================#
    # move model to device
    model.to(device)
    logger.info("Model moved to %s", device)
    
    # 0. Loop through epochs
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):

        # Set model to training mode
        model.train()
        epoch_train_loss = 0
        train_preds, train_targets = [], []
        # ======================================================#
        # 3. Set hook to training mode
        # ======================================================#
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
            epoch_train_loss += loss.item()
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(target.cpu().detach().numpy())

        # Compute training metrics
        avg_train_loss = epoch_train_loss / len(train_loader) #Avg. loss per epoch
        # Record training loss/epoch
        hook.record_tensor_value(tensor_name="train_loss", tensor_value=avg_train_loss)
        train_mse = mean_squared_error(train_targets, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Log training metrics
        logger.info("Epoch %d/%d, Training Loss: %.3f", epoch, epochs, avg_train_loss)
        logger.info("Epoch %d/%d, Training MSE: %.3f", epoch, epochs, train_mse)
        logger.info("Epoch %d/%d, Training R2: %.2f, Training RMSE: %.3f, Training MAE: %.3f", epoch, epochs, train_r2, train_rmse, train_mae)

        # Perform validation
        logger.info("Validation started.")

        # Set model to validation mode
        model.eval()
        # ======================================================#
        # 3.1 Set hook to validation mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.EVAL)
        epoch_val_loss = 0
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
                
                epoch_val_loss += loss.item()
                val_preds.extend(preds.cpu().detach().numpy())
                val_targets.extend(target.cpu().detach().numpy())

            # Compute validation metrics
            avg_val_loss = epoch_val_loss / len(val_loader) #avg. loss per epoch
            # Record validation loss
            hook.record_tensor_value(tensor_name="val_loss", tensor_value=avg_val_loss)
            val_mse = mean_squared_error(val_targets, val_preds)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(val_targets, val_preds)
            val_mae = mean_absolute_error(val_targets, val_preds)

            # Log validation metrics
            logger.info("Epoch %d/%d, Validation Loss: %.3f", epoch, epochs, avg_val_loss)
            logger.info("Epoch %d/%d, Validation MSE: %.3f", epoch, epochs, val_mse)
            logger.info("Epoch %d/%d, Validation R2: %.2f, Validation RMSE: %.3f, Validation MAE: %.3f", epoch, epochs, val_r2,
                        val_rmse, val_mae)

    # Close hook
    hook.close()
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
    model = DeepCNN(input_size=51, dropout_rate=args.dropout)

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
    parser.add_argument("--dropout", type=float, default=0.5)
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
