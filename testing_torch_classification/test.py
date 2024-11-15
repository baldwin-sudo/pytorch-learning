from typing import Iterator
import torch.nn as nn
import torch
import torch.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data as data
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from trainloop import Net
# data loading
X,Y=datasets.load_iris(return_X_y=True,as_frame=True)
# splitting into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)

# my dataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,x,y,) -> None:
        self.x=x.values # convert dataframes to values
        self.y=y.values
        super(MyIterableDataset).__init__()
    def __iter__(self) -> Iterator:
        for xi, yi in zip(self.x, self.y):
            yield torch.tensor(xi, dtype=torch.float32), torch.tensor(yi, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
# my dataloader
train_dataset = MyIterableDataset(x_train,y_train)
test_dataset = MyIterableDataset(x_test,y_test)

train_dataloader=data.DataLoader(train_dataset,batch_size=16)
test_dataloader=data.DataLoader(test_dataset,batch_size=16)
network =Net()
optimizer = optim.SGD(network.parameters(),lr=0.001,momentum=0.9,nesterov=False)
#optimizer = optim.Adam(network.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filename}")
    print(f"Resuming training from epoch {epoch}, last loss {loss}")
    return epoch, loss

# Example: Loading the checkpoint
epoch, loss = load_checkpoint(network, optimizer)

def test_loop(dataloader, network, criterion):
    network.eval()  # Set the network to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_x, batch_y_true in dataloader:
            # Forward pass
            batch_y_pred = network(batch_x)

            # Calculate loss
            loss = criterion(batch_y_pred, batch_y_true.long())
            total_loss += loss.item()

            # Calculate predictions and accuracy
            predicted_classes = batch_y_pred.argmax(dim=1)  # Get the class with highest score
            correct += (predicted_classes == batch_y_true).sum().item()
            total += batch_y_true.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
test_loop(test_dataloader,network,criterion)