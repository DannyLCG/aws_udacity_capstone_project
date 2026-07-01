'''Local inference script. Loads the trained ensemble model and predicts the
pMIC value for one or more peptide sequences, without needing a SageMaker endpoint.'''
import os
import argparse

import numpy as np
import torch

from utils.encoder import OneHotEncoder
from models.ensemble_model import CNN_biLSTM_Model

# Default path to the saved ensemble weights
MODEL_PATH = os.path.join("models", "ensemble_model.pth")

def load_model(model_path=MODEL_PATH, device="cpu"):
    '''Load the trained ensemble model from a saved state_dict.
    -------------------------------------------------
    Params:
        model_path: str, path to the saved .pth weights.
        device: str, computation device ("cpu" or "cuda").
    '''
    # Load the saved artifact (can be a full model or a state_dict)
    with open(model_path, "rb") as f:
        artifact = torch.load(f, map_location=device, weights_only=False)

    # If a full model was saved, use it directly; otherwise load the weights
    if isinstance(artifact, CNN_biLSTM_Model):
        model = artifact
    else:
        model = CNN_biLSTM_Model(input_size=51)
        model.load_state_dict(artifact)

    # Move to device and set evaluation mode
    model.to(device)
    model.eval()
    return model

def predict(sequences, model=None, device="cpu"):
    '''Predict the pMIC value for a list of peptide sequences.
    -------------------------------------------------
    Params:
        sequences: array-like, peptide sequences in single-letter amino acid code.
        model: class, a loaded model. If None (default) the saved ensemble is loaded.
        device: str, computation device.
    '''
    # Load the model if none is provided
    if model is None:
        model = load_model(device=device)

    # Encode the sequences into 20x51 matrices, same as the training pipeline
    encoder = OneHotEncoder(max_len=50, stop_signal=True)
    encoded = encoder.encode(sequences)
    inputs = torch.tensor(encoded, dtype=torch.float32).to(device)

    # Make predictions without tracking gradients
    with torch.no_grad():
        preds = model(inputs)

    # Return a flat array, one pMIC value per input sequence
    return preds.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pMIC for peptide sequences.")

    parser.add_argument("sequences", nargs="+", help="One or more peptide sequences (single-letter code).")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # Run inference and print one prediction per sequence
    model = load_model(args.model_path, args.device)
    predictions = predict(args.sequences, model=model, device=args.device)
    for seq, pmic in zip(args.sequences, predictions):
        print(f"{seq}\tpredicted pMIC: {pmic:.3f}")
