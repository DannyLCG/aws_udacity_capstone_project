import os
import torch

from models.ensemble_model import CNN_biLSTM_Model

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instance our trained mode
    model = CNN_biLSTM_Model(input_size=51)
    model.to(device)

    # Load our trained model
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))

    # Set model to evaluation mode
    model.eval()

    return model