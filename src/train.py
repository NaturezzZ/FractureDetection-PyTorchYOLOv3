import time
import torch.optim as optim
from torch.utils.data import DataLoader

from param import *
from data_loader import FractionDataset
from model import CNN_Net
from test import test

# loading training data
train_data = FractionDataset(train_data_dir)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("Loading training data: Success.")
time.sleep(0.5)

# loading test data
test_data = FractionDataset(test_data_dir)
train_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("Loading validation data: Success.")
time.sleep(0.5)

model = CNN_Net()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Start Training.")
time.sleep(0.5)
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = test(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training.')
