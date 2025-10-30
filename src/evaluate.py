import numpy as np
import re
from sentence_transformers import SentenceTransformer

class ResumeEvaluator:
    """Handles all logic for comparing and scoring a resume against a job description."""
    def __init__(self, config):
        """
        Initializes the evaluator with configuration and loads the similarity model.
        
        Args:
            config (dict): The entire loaded config.yml file.
        """
        model_name = config["models"]["similarity_model"]
        self.sim_model = SentenceTransformer(model_name)

        eval_config = config["evaluation"]
        self.semantic_weight = eval_config["weights"]["semantic"]
        self.keyword_weight = eval_config["weights"]["keyword"]
        self.hard_skill_keywords = list(set(eval_config["hard_skill_keywords"]))
        self.standalone_skills = set(eval_config["standalone_regex_skills"])

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

    def evaluate_match(self, resume_text, jd_text):
        """
        Performs a full evaluation of a single resume against a single job description.
        
        Args:
            resume_text (str): The cleaned text of the resume.
            jd_text (str): The cleaned text of the job description.
            
        Returns:
            dict: A dictionary containing all score components.
        """

        resume_vec = self.sim_model.encode(resume_text)
        jd_vec = self.sim_model.encode(jd_text)
        semantic_score = self._get_cosine_similarity(resume_vec, jd_vec)
        semantic_score_normalized = (semantic_score + 1) / 2 #normalize

        keyword_score, required, overlap = self._calculate_keyword_overlap(resume_text, jd_text)

        final_score = (semantic_score_normalized * self.semantic_weight) + (keyword_score * self.keyword_weight)
        
        return {
            "total_score": final_score,
            "semantic_score": semantic_score_normalized,
            "keyword_score": keyword_score,
            "required_skills": required,
            "overlapping_skills": overlap
        }