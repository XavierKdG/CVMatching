import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df[['resume_text','job_text']], df['label'], test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def embed_texts(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:,0,:]  # CLS token
    return embeddings.cpu()

resume_emb_train = embed_texts(X_train['resume_text'].tolist())
job_emb_train = embed_texts(X_train['job_text'].tolist())
resume_emb_test = embed_texts(X_test['resume_text'].tolist())
job_emb_test = embed_texts(X_test['job_text'].tolist())

X_train_emb = torch.abs(resume_emb_train - job_emb_train)
X_test_emb = torch.abs(resume_emb_test - job_emb_test)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

class CVJobClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

input_dim = X_train_emb.shape[1]
classifier = CVJobClassifier(input_dim).to(device)

class_counts = np.bincount(y_train)
weight_for_0 = class_counts[1] / sum(class_counts)
weight_for_1 = class_counts[0] / sum(class_counts)
weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)

criterion = nn.BCELoss()  # Kan ook met weighted loss: see advanced
optimizer = optim.Adam(classifier.parameters(), lr=2e-5)

X_train_emb, X_test_emb = X_train_emb.to(device), X_test_emb.to(device)
y_train_tensor, y_test_tensor = y_train_tensor.to(device), y_test_tensor.to(device)

epochs, batch_size = 5, 32
for epoch in range(epochs):
    classifier.train()
    permutation = torch.randperm(X_train_emb.size()[0])
    for i in range(0, X_train_emb.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_emb[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = classifier(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

classifier.eval()
with torch.no_grad():
    y_pred = classifier(X_test_emb)
    y_pred_label = (y_pred > 0.5).float()

print("Accuracy:", accuracy_score(y_test_tensor.cpu(), y_pred_label.cpu()))
print("F1-score:", f1_score(y_test_tensor.cpu(), y_pred_label.cpu()))
