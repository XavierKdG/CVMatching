import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer

try:
    torch.cuda.init()
except AssertionError:
    pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ResumeEvaluator:
    """Handles all logic for comparing and scoring a resume against a job description."""
    def __init__(self, config, model_name=None):
        """
        Initializes the evaluator with configuration and loads the similarity model.
        
        Args:
            config (dict): The entire loaded config.yml file.
            model_name (str, optional): Override the similarity model from config.
        """
        model_name = model_name or config["models"]["similarity_model"]
        self.sim_model = SentenceTransformer(model_name, device=DEVICE)

        eval_config = config["evaluation"]
        self._original_semantic_weight = eval_config["weights"]["semantic"]
        self._original_keyword_weight = eval_config["weights"]["keyword"]
        self.semantic_weight = self._original_semantic_weight
        self.keyword_weight = self._original_keyword_weight
        self.hard_skill_keywords = list(set(eval_config["hard_skill_keywords"]))
        self.standalone_skills = set(eval_config["standalone_regex_skills"])

    def set_keyword_enabled(self, enabled):
        if enabled:
            self.semantic_weight = self._original_semantic_weight
            self.keyword_weight = self._original_keyword_weight
        else:
            self.semantic_weight = 1.0
            self.keyword_weight = 0.0

    def _get_cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity (score from -1 to 1)."""
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        if np.all(vec1 == 0) or np.all(vec2 == 0): return -1.0
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def _calculate_keyword_overlap(self, resume_text, jd_text):
        """
        Calculates the keyword overlap using the config-defined lists.
        """
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        resume_skills = set()
        required_skills = set()
        
        for skill in self.hard_skill_keywords: #Resumes
            if skill in self.standalone_skills:
                pattern = r"\b" + re.escape(skill) + r"\b"
                if re.search(pattern, resume_lower):
                    resume_skills.add(skill)
            else:
                if skill in resume_lower:
                    resume_skills.add(skill)

        for skill in self.hard_skill_keywords: #Jobs
            if skill in self.standalone_skills:
                pattern = r"\b" + re.escape(skill) + r"\b"
                if re.search(pattern, jd_lower):
                    required_skills.add(skill)
            else:
                if skill in jd_lower:
                    required_skills.add(skill)

        if not required_skills:
            return 0.0, set(), set()
        
        intersection = resume_skills.intersection(required_skills)
        overlap_score = len(intersection) / len(required_skills)
        
        return overlap_score, required_skills, intersection

    def _score_from_vectors(self, resume_vec, jd_vec, resume_text, jd_text):
        semantic_score = self._get_cosine_similarity(resume_vec, jd_vec)
        semantic_score_normalized = (semantic_score + 1) / 2
        keyword_score, required, overlap = self._calculate_keyword_overlap(resume_text, jd_text)
        final_score = (semantic_score_normalized * self.semantic_weight) + (keyword_score * self.keyword_weight)
        return {
            "total_score": final_score,
            "semantic_score": semantic_score_normalized,
            "keyword_score": keyword_score,
            "required_skills": required,
            "overlapping_skills": overlap
        }

    def evaluate_match(self, resume_text, jd_text):
        resume_vec = self.sim_model.encode(resume_text)
        jd_vec = self.sim_model.encode(jd_text)
        return self._score_from_vectors(resume_vec, jd_vec, resume_text, jd_text)

    def evaluate_batch(self, resume_texts, jd_text):
        """Evaluates multiple resumes against one JD using batched GPU encoding."""
        jd_vec = self.sim_model.encode(jd_text)
        resume_vecs = self.sim_model.encode(resume_texts)
        return [
            self._score_from_vectors(resume_vecs[i], jd_vec, resume_texts[i], jd_text)
            for i in range(len(resume_texts))
        ]