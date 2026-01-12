from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm


# evalueert voor elke jobs hoe goed hij past voor elke cv en wederzijds hetzelfde. job<> cv,cv <>job
def evaluate_mutual_ranking(job_emb: list, cv_emb:list) -> dict:
    cv_to_job = util.cos_sim(cv_emb, job_emb).cpu().numpy()
    job_to_cv = util.cos_sim(job_emb, cv_emb).cpu().numpy()

    reciprocal_ranks = []

    for cv_idx in range(cv_to_job.shape[0]):
        best_job = np.argmax(cv_to_job[cv_idx])

        ranks = np.argsort(job_to_cv[best_job])[::-1]
        rank_of_cv = np.where(ranks == cv_idx)[0][0] + 1

        reciprocal_ranks.append(1.0 / rank_of_cv)

    return {
        "mean_mutual_mrr": float(np.mean(reciprocal_ranks))
    }

# kijkt naar als we een woord weghalen hoe robuust en consistent het model blijft in zijn matching grote verschil = slecht.
def perturbation_test(model, cv_texts, job_emb, remove_word=None) -> float:
    if remove_word is None:
        remove_word = ["Python"]

    all_drops = []

    for word in remove_word:
        for cv in cv_texts[:100]:  # sample om runtime te beperken
            original_emb = model.encode(cv, convert_to_tensor=True)
            perturbed_cv = cv.replace(word, "")
            perturbed_emb = model.encode(perturbed_cv, convert_to_tensor=True)

            sim_orig = util.cos_sim(original_emb, job_emb).max().item()
            sim_pert = util.cos_sim(perturbed_emb, job_emb).max().item()

            all_drops.append(sim_orig - sim_pert)

    return float(np.mean(all_drops))

# elke cv neem je de top jobs en tel die hoevaak ze voorkomen in de alle cv's.  
def evaluate_hubness(job_emb, cv_emb, top_k=5) -> dict:
    sims = util.cos_sim(cv_emb, job_emb).cpu().numpy()
    topk_indices = np.argsort(sims, axis=1)[:, ::-1][:, :top_k]

    counts = np.bincount(topk_indices.flatten(), minlength=job_emb.shape[0])

    return {
        "mean_hubness": float(np.mean(counts)),
        "max_hubness": int(np.max(counts)),
        "hubness_std": float(np.std(counts)),
    }

def evaluate_models(model_paths, job_texts, cv_texts, perturb_word=None, batch_size=64, top_k_hub=5) -> dict:
    """
    Evalueer meerdere modellen op:
    - Mutual ranking
    - Perturbation robustness
    - Hubness

    Args:
        model_paths (list[str]): pad of naam van modellen
        job_texts (list[str]): lijst van job descriptions
        cv_texts (list[str]): lijst van CV teksten
        perturb_word (str): woord dat weggehaald wordt voor perturbation test
        batch_size (int)
        top_k_hub (int): top-k jobs voor hubness

    Returns:
        dict: resultaten per model
    """
    if perturb_word is None:
        perturb_word = "Python"

    results = {}

    for model_path in model_paths:
        print("\n" + "="*50)
        print(f"Evaluating model: {model_path}")
        print("="*50)

        model = SentenceTransformer(model_path)

        job_emb = model.encode(job_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        cv_emb  = model.encode(cv_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)


        mr = evaluate_mutual_ranking(job_emb, cv_emb)
        print(f"Mean Mutual MRR : {mr['mean_mutual_mrr']:.4f}")

        pt = perturbation_test(model, cv_texts, job_emb, remove_word=perturb_word)
        print(f"Perturbation Drop ({perturb_word}) : {pt:.4f}")

        hub = evaluate_hubness(job_emb, cv_emb, top_k=top_k_hub)
        print(f"Mean Hubness : {hub['mean_hubness']:.2f}, Max Hubness : {hub['max_hubness']}, Hub Std : {hub['hubness_std']:.2f}")

        results[model_path] = {
            "mutual_ranking": mr,
            "perturbation_drop": pt,
            "hubness": hub
        }
    return results

if __name__ == "__main__":
    model_paths = [
    "all-MiniLM-L6-v2",
    "./models/tsdae_model_fast",
    "./models/tsdae_model_full"
    ]

    jobs_df = pd.read_csv("./data/processed/job_descriptions2_cleaned.csv")
    cvs_df  = pd.read_csv("./data/processed/Resume_cleaned.csv")

    job_texts = jobs_df["Job Description"].astype(str).tolist()
    cv_texts  = cvs_df["Resume_str"].astype(str).tolist()


    keywords = ["Python", "SQL", "Excel", "Java"]
    results = evaluate_models(
        model_paths=model_paths,
        job_texts=job_texts,
        cv_texts=cv_texts,
        perturb_word=keywords,
        batch_size=64,
        top_k_hub=5
    )

    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)