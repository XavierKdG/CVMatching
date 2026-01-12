from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import List

# -----------------------------
# Dataset
# -----------------------------
class TextPairsDataset(Dataset):
    """Custom Dataset voor TSDAE: (noised_doc, teacher_embedding)"""
    def __init__(self, noised_docs: List[str], teacher_embeddings: torch.Tensor):
        self.noised_docs = noised_docs
        self.teacher_embeddings = teacher_embeddings

    def __len__(self):
        return len(self.noised_docs)

    def __getitem__(self, idx):
        return self.noised_docs[idx], self.teacher_embeddings[idx]

# -----------------------------
# TSDAE Trainer
# -----------------------------
class TSDAETrainer:
    def __init__(self, model_name, batch_size=32, lr=1e-5, epochs=3, save_dir="./models/tsdae_model_fast"):
        self.encoder = SentenceTransformer(model_name)
        self.tokenizer = self.encoder.tokenizer
        self.teacher = SentenceTransformer(model_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)
        self.teacher.to(self.device)
        os.makedirs(self.save_dir, exist_ok=True)

    # -----------------------------
    # Add noise
    # -----------------------------
    def add_noise(self, documents: List[str], noise_level: float = 0.1) -> List[str]:
        noised_docs = []
        mask_token = self.tokenizer.mask_token

        for doc in documents:
            tokens = doc.split()
            n = len(tokens)
            if n == 0:
                noised_docs.append(doc)
                continue

            # Mask
            n_mask = max(1, int(n * noise_level))
            mask_indices = torch.randperm(n)[:n_mask].tolist()
            for idx in mask_indices:
                tokens[idx] = mask_token

            # Delete
            n_delete = max(1, int(n * noise_level))
            delete_indices = set(torch.randperm(n)[:n_delete].tolist())
            tokens = [tok for i, tok in enumerate(tokens) if i not in delete_indices]
            if len(tokens) == 0:
                tokens = doc.split()

            # Local shuffle
            window = 3
            tokens_shuffled = tokens.copy()
            for i in range(len(tokens_shuffled)):
                start = max(0, i - window)
                end = min(len(tokens_shuffled), i + window + 1)
                j = int(torch.randint(start, end, (1,)).item())
                tokens_shuffled[i], tokens_shuffled[j] = tokens_shuffled[j], tokens_shuffled[i]

            noised_docs.append(" ".join(tokens_shuffled))
        return noised_docs

    # -----------------------------
    # Train
    # -----------------------------
    def train(self, train_docs: List[str], val_docs: List[str], noise_level: float = 0.1):
        # Precompute teacher embeddings
        with torch.no_grad():
            teacher_embeddings = self.teacher.encode(train_docs, batch_size=self.batch_size, convert_to_tensor=True, device=self.device)

        # Add noise
        noised_docs = self.add_noise(train_docs, noise_level=noise_level)

        # Dataset & dataloader
        dataset = TextPairsDataset(noised_docs, teacher_embeddings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        best_val_score = -1.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for noised_batch, teacher_emb_batch in dataloader:
                optimizer.zero_grad()

                # Tokenize student batch
                noised_inputs = self.encoder.tokenize(noised_batch)
                noised_inputs = {k: v.to(self.device) for k, v in noised_inputs.items()}

                # Forward pass (student)
                student_out = self.encoder(noised_inputs)
                student_embs = student_out["sentence_embedding"]

                # Teacher embeddings (already on device)
                teacher_emb_batch = teacher_emb_batch.to(self.device)

                # Loss
                loss = 1 - F.cosine_similarity(student_embs, teacher_emb_batch, dim=1).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Validation (batched)
            val_score = self.evaluate(val_docs, noise_level=0.0)
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Val CosSim: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                self.encoder.save(self.save_dir)
                print(f"Model opgeslagen (beste val_score={best_val_score:.4f})")

        print("Training compleet.")

    # -----------------------------
    # Evaluate
    # -----------------------------
    def evaluate(self, docs: List[str], noise_level: float = 0.0) -> float:
        self.encoder.eval()
        noised_docs = self.add_noise(docs, noise_level=noise_level)
        cosine_sims = []

        with torch.no_grad():
            # Batched embeddings
            student_embs = self.encoder.encode(noised_docs, batch_size=self.batch_size, convert_to_tensor=True, device=self.device)
            teacher_embs = self.teacher.encode(docs, batch_size=self.batch_size, convert_to_tensor=True, device=self.device)

            sims = F.cosine_similarity(student_embs, teacher_embs, dim=1)
            cosine_sims = sims.cpu().tolist()

        self.encoder.train()
        return sum(cosine_sims) / len(cosine_sims)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    dataset = pd.read_csv("./data/processed/job_descriptions2_cleaned.csv")
    original_docs = dataset['Job Description'].tolist()

    # Train/Val/Test split
    train_docs, temp_docs = train_test_split(original_docs, test_size=0.3, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}")

    # Trainer
    trainer = TSDAETrainer(
        model_name="all-MiniLM-L6-v2",
        batch_size=64,
        lr=1e-5,
        epochs=5,
        save_dir="./models/tsdae_model_full"
    )

    trainer.train(train_docs, val_docs, noise_level=0.15)

    test_score = trainer.evaluate(test_docs, noise_level=0.0)
    print(f"Test Cosine Similarity: {test_score:.4f}")
