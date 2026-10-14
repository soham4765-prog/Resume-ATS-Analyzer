
import re

# Canonical skill name: accepted variations
SKILL_ALIASES = {
    "python": ["python", "python3"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": [
        "natural language processing", "nlp"
    ],
    "computer vision": ["computer vision"],
    "generative ai": ["generative ai", "gen ai"],
    "large language models": [
        "large language models", "large language model", "llms", "llm"
    ],
    "retrieval augmented generation": [
        "retrieval augmented generation", "rag"
    ],
    "transformers": ["transformers", "transformer models"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "opencv": ["opencv", "open cv"],
    "yolo": ["yolo", "yolov8", "yolo11"],
    "sql": ["sql", "mysql", "postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"],
    "git": ["git", "github"],
    "docker": ["docker"],
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "streamlit": ["streamlit"],
    "rest api": ["rest api", "restful api"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "java": ["java"],
    "c++": ["c++"],
    "linux": ["linux", "ubuntu"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel"],
}


def _contains_term(text: str, term: str) -> bool:
    """Match a whole word or phrase, ignoring letter case."""
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text: str) -> set[str]:
    """Return canonical skills detected in a document."""
    if not text:
        return set()

    found = set()

    for canonical, aliases in SKILL_ALIASES.items():
        if any(_contains_term(text, alias) for alias in aliases):
            found.add(canonical)

    return found


def compare_skills(
    resume_text: str,
    job_description: str
) -> dict:
    """Compare resume skills against skills found in the job description."""

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)

    score = (
        round(len(matched) / len(job_skills) * 100, 2)
        if job_skills else 0.0
    )

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "keyword_score": score,
        "job_skills_found": len(job_skills),
        "matched_count": len(matched),
    }
