from sentence_transformers import (SentenceTransformer, losses,util,datasets)
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
import nltk


# Download tokenizer
nltk.download("punkt_tab")

# CV ↔ Job Retrieval Evaluatie
def retrieval_evaluate(model: SentenceTransformer, cvs: list, jobs: list, k=5) -> float:

    print("\nRunning retrieval evaluation...")

    cv_emb = model.encode(
        cvs,
        batch_size=64,
        convert_to_tensor=True
    )

    job_emb = model.encode(
        jobs,
        batch_size=64,
        convert_to_tensor=True
    )

    scores = util.cos_sim(cv_emb, job_emb)

    # Top-K scores per CV
    topk = scores.topk(k, dim=1).values

    avg_score = topk.mean().item()

    print(f"Average Top-{k} Cosine Similarity: {avg_score:.4f}")

    return avg_score


# Main

if __name__ == "__main__":

    dataset = pd.read_csv("./data/processed/job_descriptions2_cleaned.csv")

    texts = dataset["Job Description"].dropna().tolist()

    train_docs, temp_docs = train_test_split(
        texts, test_size=0.3, random_state=42
    )

    val_docs, test_docs = train_test_split(
        temp_docs, test_size=0.5, random_state=42
    )

    print(f"Train: {len(train_docs)}")
    print(f"Val:   {len(val_docs)}")
    print(f"Test:  {len(test_docs)}")


    # Model
    model = SentenceTransformer("all-MiniLM-L6-v2")



    # TSDAE Dataset
    train_dataset = datasets.DenoisingAutoEncoderDataset(
        sentences=train_docs
    )


    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=64,
        drop_last=True
    )

    # TSDAE Loss
    train_loss = losses.DenoisingAutoEncoderLoss(model)

    # Output map
    output_dir = "./models/tsdae_model_last"
    os.makedirs(output_dir, exist_ok=True)

    # Training
    print("\nTraining...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],

        epochs=1,          
        warmup_steps=500,

        optimizer_params={"lr": 2e-5},

        output_path=output_dir,

        show_progress_bar=True
    )


    # Retrieval evaluatie
    print("\nEvaluating retrieval...")

    # Simuleer CVs en Jobs
    mid = len(test_docs) // 2

    cvs = test_docs[:mid]
    jobs = test_docs[mid:]

    retrieval_evaluate(model, cvs, jobs, k=5)
