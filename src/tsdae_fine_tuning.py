from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

# Helper class om per index een (noised_sentence, original_sentence) paar te leveren
class TextPairsDataset(Dataset):
    """
    Custom PyTorch Dataset voor TSDAE-training.

    Slaat noised en originele documenten op en geeft per index een tuple terug:
    (noised_doc, original_doc).

    Attributes:
        noised_docs (List[str]): De documenten met ruis.
        original_docs (List[str]): De originele documenten zonder ruis.
    """
    def __init__(self, noised_docs, original_docs):
        self.noised_docs = noised_docs
        self.original_docs = original_docs

    def __len__(self):
        return len(self.noised_docs)

    def __getitem__(self, idx):
        return self.noised_docs[idx], self.original_docs[idx]

# het traint dus geen embeddings, maar een model dat beter embeddings kan genereren door zinnen beter te reconstrueren.
# het doel van dit is om de sentence transformer model te trainen, de weights van het model worden getrained.
# op basis van de toegevoegde ruis aan de originele documenten.
class TSDAETrainer:
    def __init__(self, model_name, batch_size=16, lr=1e-5, epochs=3, save_dir="./tsdae_model"):
        
        # 1. Load encoder
        self.encoder = SentenceTransformer(model_name) 
        #load tokenizer from encoder
        self.tokenizer = self.encoder.tokenizer  

        # 2. Store hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir

        # 3. Device (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)

        # we maken hier een functie die ruis toevoegd aan de originele document, dus de originele dataset.
    def add_noise(self, documents: list[str], noise_level: float = 0.1) -> list[str]:
            """
            TSDAE-style  noise toevoegen  aan document : masking + deletion + zin shuffle.
            noise_level = Hoeveel van de tokens % gaat beinvloedt worden, dus hoe groot wordt de noise. (e.g., 0.1 = 10%).
            """
            noised_docs = []
            mask_token : str = self.tokenizer.mask_token

            # split het document in tokens
            for doc in documents:
                tokens = doc.split()
                n = len(tokens)
                if n == 0:
                    noised_docs.append(doc)
                    continue

                # Maskeren van de tokens berekend hoeveel % en daarna wordt het random gekozen indices vervangen door de mask token.
                n_mask = max(1, int(n * noise_level))
                mask_indices = torch.randperm(n)[:n_mask].tolist()
                for idx in mask_indices:
                    tokens[idx] = mask_token

                # Verwijderen van de tokens uit de originele zinnen om ruis te gebven
                n_delete = max(1, int(n * noise_level))
                delete_indices = set(torch.randperm(n)[:n_delete].tolist())    
                tokens_after_delete = [
                    tok for i, tok in enumerate(tokens) if i not in delete_indices
                ]
                if len(tokens_after_delete) == 0:
                    tokens_after_delete = tokens

                # Zin shuffle bv zin is "de kat zit op de mat"
                # na lokale shuffle kan het worden "de op zit kat de mat" dus het shuffled posities binnen een bepaalde window
                window = 3  
                tokens_shuffled: list[str] = tokens.copy()
                for i in range(len(tokens_shuffled)):
                    start = max(0, i - window)
                    end = min(len(tokens_shuffled), i + window + 1)

                    j:int = int(torch.randint(start, end, (1,)).item()) 
                    tokens_shuffled[i], tokens_shuffled[j] = tokens_shuffled[j], tokens_shuffled[i]
                    
                noised_docs.append(" ".join(tokens_shuffled))

            return noised_docs

        # het trainen van het model met de originele documenten en toegevoegde ruis.
    def train(self, original_doc: list[str], noise_level: float = 0.1) -> None:
            """
            Train TSDAE model met de originele documenten en toegevoegde ruis.
            """
            # maak kopie van originele documenten
            original_doc2 : List[str]= original_doc.copy()
            # voeg ruis toe aan de originele documenten
            noised_docs : List[str]  = self.add_noise(original_doc, noise_level)

            # een tuple van [str,str]
            tsdae_dataset : TextPairsDataset= TextPairsDataset(noised_docs=noised_docs, original_docs=original_doc2)

            dataloader = DataLoader(
                    dataset = tsdae_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,           # shuffle tijdens training
                    drop_last=True,         # laatste batch droppen als die kleiner is dan batch_size
                )

            optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            for epoch in range (self.epochs):
                for noised_batch, original_batch in dataloader:
                    # reset gradients
                    optimizer.zero_grad()
                    # verkregen embeddings van de noised zinnen
                    reconstructed_embeddings : torch.Tensor = self.encoder(noised_batch)  
                    # target embeddings van originele zinnen
                    with torch.no_grad():
                        target_embeddings : torch.Tensor = self.encoder(original_batch)
                    #bereken loss op basis van cosine similarity    
                    loss = 1 - F.cosine_similarity(reconstructed_embeddings, target_embeddings, dim=1).mean()
                    #backpropogation
                    loss.backward()
                    #update weights
                    optimizer.step()

            self.encoder.save(self.save_dir)
            print(f"Model succesvol opgeslagen op: {self.save_dir}")

if __name__ == "__main__":
    
    trainer = TSDAETrainer(model_name="all-MiniLM-L6-v2", batch_size=16, lr=1e-5, epochs=3, save_dir="./tsdae_model")







  

