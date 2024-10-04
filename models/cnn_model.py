from torch import nn
from torch.nn.functional import functional as F

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
        self.fc1 = nn.Linear(256 * 36, 512)
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

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        output = self.fc3(x) #For regression return the direct output wo. any activation function

        return output
