import numpy as np
from tqdm import tqdm  # als je progress bars wilt bij embedden
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split


X_train_tensor = torch.tensor(X_train_emb, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_emb, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor, X_test_tensor = X_train_tensor.to(device), X_test_tensor.to(device)
y_train_tensor, y_test_tensor = y_train_tensor.to(device), y_test_tensor.to(device)

class CVJobMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output logit
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]
model = CVJobMLP(input_dim).to(device)

criterion = nn.BCEWithLogitsLoss()  # numeriek stabieler
optimizer = optim.Adam(model.parameters(), lr=1e-3)


epochs = 20
batch_size = 32

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    total_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(X_train_tensor) / batch_size)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    y_pred = (probs > 0.5).float()

acc = accuracy_score(y_test_tensor.cpu(), y_pred.cpu())
f1 = f1_score(y_test_tensor.cpu(), y_pred.cpu())

print(f"\nEvaluatie resultaten:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
