# compare_models.py
import pandas as pd
from logger import setup_logger

# 1️⃣ Setup logger
logger = setup_logger(name="GlobalLogger", log_file="./logs/all_processes.log")


def compare_models(
        doc2vec_csv="./data/processed/doc2vec_job_resume_matches.csv",
        minilm_csv="./data/processed/minilm_job_resume_matches.csv",
        jobs_csv="./data/processed/job_descriptions2cleaned.csv",
        resumes_csv="./data/raw/Resume.csv",
        output_csv="./data/processed/model_comparison_detailed.csv"
    ):
    logger.info("Loading jobs and resumes...")
    jobs = pd.read_csv(jobs_csv).reset_index().rename(columns={"index": "Job_Index"})
    resumes = pd.read_csv(resumes_csv).reset_index().rename(columns={"index": "Resume_Index"})

    logger.info("Loading model match CSVs...")
    df_doc2vec = pd.read_csv(doc2vec_csv)
    df_minilm = pd.read_csv(minilm_csv)

    logger.info("Merging job information into model results...")
    df_doc2vec = pd.merge(df_doc2vec, jobs[["Job_Index", "Job ID", "Job Category"]],
                          on="Job_Index", how="left")
    df_minilm = pd.merge(df_minilm, jobs[["Job_Index", "Job ID", "Job Category"]],
                         on="Job_Index", how="left")

    logger.info("Merging resume categories...")
    df_doc2vec = pd.merge(df_doc2vec, resumes[["Resume_Index", "Category"]],
                          left_on="Best_Resume_Index", right_on="Resume_Index", how="left")
    df_minilm = pd.merge(df_minilm, resumes[["Resume_Index", "Category"]],
                         left_on="Best_Resume_Index", right_on="Resume_Index", how="left")

    # Rename columns
    df_doc2vec = df_doc2vec.rename(columns={"Category": "Doc2Vec_Resume_Category",
                                            "Similarity": "Doc2Vec_Similarity"})
    df_minilm = df_minilm.rename(columns={"Category": "MiniLM_Resume_Category",
                                          "Similarity": "MiniLM_Similarity"})

    # Ensure same length/order
    n = min(len(df_doc2vec), len(df_minilm))
    df_doc2vec = df_doc2vec.iloc[:n].reset_index(drop=True)
    df_minilm = df_minilm.iloc[:n].reset_index(drop=True)

    # Combine results
    df_compare = pd.DataFrame({
        "Job ID": df_doc2vec["Job ID"],
        "Job Category": df_doc2vec["Job Category"],
        "Doc2Vec_Similarity": df_doc2vec["Doc2Vec_Similarity"],
        "Doc2Vec_Resume_Category": df_doc2vec["Doc2Vec_Resume_Category"],
        "MiniLM_Similarity": df_minilm["MiniLM_Similarity"],
        "MiniLM_Resume_Category": df_minilm["MiniLM_Resume_Category"]
    })

    df_compare["Better_Model"] = df_compare.apply(
        lambda row: "Doc2Vec" if row["Doc2Vec_Similarity"] > row["MiniLM_Similarity"] else "MiniLM",
        axis=1
    )

    # Save CSV
    df_compare.to_csv(output_csv, index=False)
    logger.info(f"Saved detailed comparison CSV at: {output_csv}")

    # Print summary
    avg_doc2vec = df_compare["Doc2Vec_Similarity"].mean()
    avg_minilm = df_compare["MiniLM_Similarity"].mean()
    logger.info(f"Average best-match similarity - Doc2Vec: {avg_doc2vec:.4f}")
    logger.info(f"Average best-match similarity - MiniLM: {avg_minilm:.4f}")
    winner = "Doc2Vec" if avg_doc2vec > avg_minilm else "MiniLM"
    logger.info(f"Overall better model: {winner}")


if __name__ == "__main__":
    compare_models()
