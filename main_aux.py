import torch
from model import AuxDrop_MLP
from dataset import AdultDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# Custom dataset class to load preprocessed data from CSV
train_dataset = AdultDataset('adult_income_dataset/adult_train.csv')
test_dataset = AdultDataset('adult_income_dataset/adult_test.csv')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_aux = train_dataset.x_aux.shape[1]
num_base = train_dataset.x_base.shape[1]


def train_auxdrop(model, train_loader, criterion, optimizer, device, dropout_ratio):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for _, x_base, x_aux, labels in tqdm(train_loader, desc="Training AuxDrop_MLP"):
        x_base, x_aux, labels = x_base.to(device), x_aux.to(device), labels.to(device)
        optimizer.zero_grad()
        # Pass both x_base and x_aux to the model
        outputs = model(x_base, x_aux, dropout_ratio=dropout_ratio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).detach().cpu()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu())

    avg_loss = running_loss / len(train_loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    print(f"Average Training Loss: {avg_loss:.4f}, Average Training Accuracy: {avg_acc:.4f}")

def test_auxdrop(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _, x_base, x_aux, labels in tqdm(test_loader, desc="Testing AuxDrop_MLP"):
            x_base, x_aux, labels = x_base.to(device), x_aux.to(device), labels.to(device)
            # Pass both x_base and x_aux to the model
            outputs = model(x_base, x_aux, dropout_ratio=0)  # Set dropout_ratio to 0 during testing
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy (AuxDrop_MLP): {accuracy:.2f}%")
    return accuracy

# Initialize model (AuxDrop_MLP)
model = AuxDrop_MLP(input_size=num_base, hidden_sizes=[128, 64, 32, 16], output_size=2, aux_layer_idx=2,
                    total_features=num_base+num_aux, num_aux_features=num_aux)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Set dropout ratio (you can experiment with different values)
dropout_ratio = 0.3

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train and test the model
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    train_auxdrop(model, train_loader, criterion, optimizer, device, dropout_ratio)
test_auxdrop(model, test_loader, device)
