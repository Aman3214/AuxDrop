import torch
from model import MLP
from dataset import AdultDataset
import torch
from torch.utils.data import DataLoader
import tqdm

# Custom dataset class to load preprocessed data from CSV
train_dataset = AdultDataset('adult_income_dataset/adult_train.csv')
test_dataset = AdultDataset('adult_income_dataset/adult_test.csv')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train_mlp(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features,_,_, labels in tqdm(train_loader, desc="Training MLP"):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")

def test_mlp(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing MLP"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy (MLP): {accuracy:.2f}%")
    return accuracy

# Initialize model (MLP or AuxDrop_MLP)
model = MLP(input_size=..., hidden_sizes=[..., ...], output_size=...)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train and test the model
train_mlp(model, train_loader, criterion, optimizer)
test_mlp(model, test_loader)
