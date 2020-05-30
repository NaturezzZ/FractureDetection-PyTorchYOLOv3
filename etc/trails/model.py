# a simple CNN
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Net(nn.Module):

    def __init__(self):
        super(CNN_Net, self).__init__()
        # input: batch_size * 3 (RGB) * 256 * 256
        # output: batch_size * 4
        
        # convolutional layers
        # 3 input image channel, 10 output channels, 5*5 square convolution, paddings=(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5, padding=(2, 2)) # 6 * 128 * 128 after pooling
        self.conv2 = nn.Conv2d(6, 16, 5, padding=(2, 2)) # 16 * 64 * 64 after pooling
        self.conv3 = nn.Conv2d(16, 40, 5, padding=(2, 2)) # 40 * 32 * 32 after pooling
        self.conv4 = nn.Conv2d(40, 100, 5, padding=(2, 2))  # 100 * 16 * 16 after pooling
        
        # fully connected layers: y = Wx + b
        self.fc1 = nn.Linear(100 * 16 * 16, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # pooling here
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 100 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = CNN_Net()
    print(net)
    
    input = torch.randn(1, 3, 256, 256)
    out = net(input)
    print(out)