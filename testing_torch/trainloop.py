import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Iterator

# Data loading
X, Y = datasets.load_iris(return_X_y=True, as_frame=True)

# Splitting into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)

# Custom dataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, x, y) -> None:
        self.x = x.values  # Convert dataframes to values
        self.y = y.values
        super(MyIterableDataset, self).__init__()

    def __iter__(self) -> Iterator:
        for xi, yi in zip(self.x, self.y):
            yield torch.tensor(xi, dtype=torch.float32), torch.tensor(yi, dtype=torch.long)

    def __len__(self):
        return len(self.x)

# My dataloader
train_dataset = MyIterableDataset(x_train, y_train)
test_dataset = MyIterableDataset(x_test, y_test)

train_dataloader = data.DataLoader(train_dataset, batch_size=16)
test_dataloader = data.DataLoader(test_dataset, batch_size=16)

# Model definition
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Linear(4, 4)
        self.dense2 = nn.Linear(4, 8)
        self.output = nn.Linear(8, 3)

        # Initialize biases to zero
        nn.init.zeros_(self.dense1.bias)
        nn.init.zeros_(self.dense2.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = nn.functional.relu(self.dense1(x))
        x = nn.functional.relu(self.dense2(x))
        x = self.output(x)  # No softmax here, CrossEntropyLoss will apply it internally
        return x

network = Net()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

epochs = 1001

# Training loop
def train_loop(dataloader, network, optimizer, criterion, epochs):
    total_loss=0
    for epoch in range(epochs):
        total_loss = 0
        for i, (batch_x, batch_y_true) in enumerate(dataloader):
            optimizer.zero_grad()  # Reset gradients to 0

            batch_y_pred = network(batch_x)  # Forward pass
            loss = criterion(batch_y_pred, batch_y_true)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
    return total_loss/len(dataloader)
# Main execution block
if __name__ == "__main__":
    total_loss=train_loop(train_dataloader, network, optimizer, criterion, epochs)

    # Save checkpoint
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "loss":total_loss
    }
    torch.save(checkpoint, "checkpoint.pth")
    print("Checkpoint saved")
