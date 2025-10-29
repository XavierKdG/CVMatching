# scripts/fine_tuning_minilm.py
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
import numpy as np

# ✅ Load base MiniLM model
model = SentenceTransformer("./models/minilm_model")

base_model = model._first_module().auto_model  # Get the raw Hugging Face transformer

# ✅ Create LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"],  # common for attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

# ✅ Wrap base transformer with LoRA
base_model = get_peft_model(base_model, lora_config)

# ✅ Put it back into SentenceTransformer
model._first_module().auto_model = base_model

# ✅ Load your data
df = pd.read_csv("./data/processed/minilm_job_resume_matches.csv")
jobs = pd.read_csv("./data/processed/job_descriptions2cleaned.csv")
resumes = pd.read_csv("./data/raw/Resume.csv")

# ✅ Prepare examples
examples = []
for _, row in df.iterrows():
    job_text = str(jobs.loc[row["Job_Index"], "Job Description"])
    resume_text = str(resumes.loc[row["Best_Resume_Index"], "Resume_str"])
    score = float(row["Similarity"])
    examples.append(InputExample(texts=[job_text, resume_text], label=score))

train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# ✅ Fine-tune MiniLM with LoRA applied
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    show_progress_bar=True
)

# ✅ Save fine-tuned LoRA MiniLM
model.save("./models/minilm_finetuned_lora")
print("✅ Fine-tuned MiniLM with LoRA saved at ./models/minilm_finetuned_lora")