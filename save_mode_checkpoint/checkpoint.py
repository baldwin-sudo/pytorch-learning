import torch
import torch.optim as optim

# Example model, optimizer, and criterion
model = torch.nn.Linear(10, 5)  # Example model
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer
criterion = torch.nn.MSELoss()  # Example loss function

# Example of training loop variables
epoch = 10  # For example, after 10 epochs

# Create a dictionary to hold the model, optimizer, and other states
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion': criterion,  # It's better to store the loss function (as a reference)
    'learning_rate': 0.001,  # Learning rate (can store other training hyperparameters)
}

# Save the checkpoint to a file
torch.save(checkpoint, 'checkpoint.pth')


# Load the checkpoint from the file

checkpoint = torch.load('checkpoint.pth')

# Restore the model, optimizer, and other parameters
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = checkpoint['criterion']  # If you saved the criterion

# Restore other parameters (e.g., epoch, learning rate)
epoch = checkpoint['epoch']
learning_rate = checkpoint['learning_rate']

print("Checkpoint loaded and training resumed!")

print("Checkpoint saved!")
