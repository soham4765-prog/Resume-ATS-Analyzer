import re
from typing import Any

from src.matching.skill_matcher import extract_skills

KEYWORD_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.40
IMPORTANCE_WEIGHTS = {"Mandatory": 3.0, "Standard": 2.0, "Preferred": 1.0}
SEMANTIC_FLOOR = 0.30
SEMANTIC_TARGET = 0.75

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "candidate",
    "experience", "familiarity", "for", "from", "good", "have", "in", "into",
    "is", "it", "knowledge", "of", "on", "or", "our", "role", "should",
    "that", "the", "their", "this", "to", "understanding", "using", "with",
    "work", "working", "you", "your", "ability", "able", "will", "would",
    "including", "such", "than", "then", "through", "where", "when", "while",
    "help", "support", "build", "develop", "use", "related", "required",
    "preferred", "qualification", "qualifications", "responsibility",
    "responsibilities", "team", "teams", "intern", "internship", "company",
    "looking", "seeking", "curious", "practical", "relevant", "basic",
    "strong", "excellent", "hands", "one", "least",
}
GENERIC_TERMS = {
    "ai", "computer", "data", "science", "model", "models", "score", "results",
    "based", "application", "resume", "candidate", "intern", "company",
    "experience", "knowledge", "skills", "skill", "work", "working",
    "project", "projects", "use", "using", "build", "develop", "support",
    "analysis", "analyze", "evaluate", "evaluation", "required", "preferred",
}

DEGREE_TERMS = {
    "computer science", "electronics", "data science", "artificial intelligence",
    "information technology", "software engineering", "electrical engineering",
}


def classify_importance(requirement: str) -> str:
    text = requirement.lower()
    if re.search(r"\b(not required|not mandatory|not essential|not necessary)\b", text):
        return "Standard"
    if re.search(
        r"\b(must|required|mandatory|essential|minimum requirement|required to|need to|needs to)\b",
        text,
    ):
        return "Mandatory"
    if re.search(
        r"\b(preferred|preferably|nice to have|bonus|desirable|ideally|optional|would be an advantage|a plus)\b",
        text,
    ):
        return "Preferred"
    # "at least one" indicates a threshold, but is not by itself a mandatory marker.
    return "Standard"


def content_tokens(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", text)
        if len(token) > 1
    }
    return tokens - STOPWORDS - GENERIC_TERMS


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.]+", " ", text.lower())).strip()


def _degree_match(requirement: str, resume_text: str) -> bool:
    req = requirement.lower()
    resume = resume_text.lower()
    is_degree_requirement = any(term in req for term in (
        "degree", "bachelor", "b.s.", "bsc", "undergraduate", "graduate",
    ))
    if not is_degree_requirement:
        return False
    req_fields = [term for term in DEGREE_TERMS if term in req]
    if not req_fields:
        return False
    resume_has_degree = any(term in resume for term in (
        "pursuing", "bachelor", "b.s.", "bsc", "undergraduate", "degree", "expected",
    ))
    return resume_has_degree and any(term in resume for term in req_fields)


def _best_resume_evidence(requirement: str, resume_text: str, fallback: str = "") -> str:
    """Choose a short, relevant resume sentence/bullet instead of a whole section."""
    chunks = re.split(r"(?<=[.!?])\s+|\n+|(?<=•)\s*", resume_text)
    chunks = [re.sub(r"\s+", " ", chunk).strip(" •-\t") for chunk in chunks]
    chunks = [chunk for chunk in chunks if len(chunk) >= 25]
    req_skills = set(extract_skills(requirement))
    req_tokens = content_tokens(requirement)
    best_chunk = ""
    best_score = -1.0

    for chunk in chunks:
        chunk_skills = set(extract_skills(chunk))
        skill_score = (
            len(req_skills & chunk_skills) / len(req_skills)
            if req_skills else 0.0
        )
        chunk_tokens = content_tokens(chunk)
        token_score = (
            len(req_tokens & chunk_tokens) / len(req_tokens)
            if req_tokens else 0.0
        )
        score = (0.75 * skill_score + 0.25 * token_score) if req_skills else token_score
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk and best_score > 0:
        return best_chunk[:500]
    if fallback:
        fallback = re.sub(r"\s+", " ", fallback).strip()
        return fallback[:500]
    return "No clear supporting resume passage identified."


def calculate_keyword_evidence(requirement: str, resume_text: str) -> tuple[float, str]:
    """Score explicit skill evidence, then meaningful content overlap and equivalences."""
    req_skills = set(extract_skills(requirement))
    resume_skills = set(extract_skills(resume_text))
    skill_matches = sorted(req_skills & resume_skills)

    if _degree_match(requirement, resume_text):
        return 100.0, "Degree field and current/enrolled study evidence matched"

    if req_skills:
        ratio = len(skill_matches) / len(req_skills)
        evidence = (
            "Matched skills: " + ", ".join(skill_matches)
            if skill_matches
            else "No required dictionary skills detected in resume"
        )
        return round(ratio * 100, 2), evidence

    req_tokens = content_tokens(requirement)
    resume_tokens = content_tokens(resume_text)
    if not req_tokens:
        return 0.0, "No meaningful keywords identified in this requirement"

    matched_tokens = sorted(req_tokens & resume_tokens)
    # Recall-style coverage is more interpretable than dividing by all resume words.
    ratio = len(matched_tokens) / len(req_tokens)
    evidence = (
        "Matched terms: " + ", ".join(matched_tokens)
        if matched_tokens
        else "No meaningful exact content-word overlap detected"
    )
    return round(ratio * 100, 2), evidence


def normalize_semantic_similarity(similarity: float) -> float:
    """Map raw cosine similarity to a bounded heuristic score."""
    normalized = (float(similarity) - SEMANTIC_FLOOR) / (SEMANTIC_TARGET - SEMANTIC_FLOOR)
    return round(max(0.0, min(1.0, normalized)) * 100, 2)


def calculate_hybrid_score(
    resume_text: str,
    keyword_result: dict,
    semantic_result: dict,
) -> dict:
    comparisons = semantic_result.get("comparisons", [])
    requirement_results = []
    weighted_total = 0.0
    total_weight = 0.0

    for item in comparisons:
        requirement = str(item.get("requirement", "")).strip()
        if not requirement:
            continue

        raw_evidence = str(item.get("resume_evidence", ""))
        similarity = float(item.get("similarity", 0.0))
        importance = classify_importance(requirement)
        importance_weight = IMPORTANCE_WEIGHTS[importance]

        keyword_score, keyword_evidence = calculate_keyword_evidence(requirement, resume_text)
        semantic_score = normalize_semantic_similarity(similarity)
        hybrid_score = KEYWORD_WEIGHT * keyword_score + SEMANTIC_WEIGHT * semantic_score

        weighted_total += hybrid_score * importance_weight
        total_weight += importance_weight

        requirement_results.append({
            "requirement": requirement,
            "importance": importance,
            "importance_weight": importance_weight,
            "keyword_score": keyword_score,
            "semantic_similarity": similarity,
            "semantic_score": semantic_score,
            "hybrid_score": round(hybrid_score, 2),
            "keyword_evidence": keyword_evidence,
            "resume_evidence": _best_resume_evidence(requirement, resume_text, raw_evidence),
        })

    overall = weighted_total / total_weight if total_weight else 0.0
    requirement_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return {
        "hybrid_score": round(overall, 2),
        "requirements_scored": len(requirement_results),
        "requirement_results": requirement_results,
        "keyword_weight": KEYWORD_WEIGHT,
        "semantic_weight": SEMANTIC_WEIGHT,
        "importance_weights": IMPORTANCE_WEIGHTS.copy(),
        "global_keyword_score": float(keyword_result.get("keyword_score", 0.0)),
        "average_cosine_similarity": float(semantic_result.get("average_similarity", 0.0)),
        "scoring_note": (
            "Prototype heuristic only. Explicit skill matches, meaningful word overlap, "
            "degree equivalence, and normalized cosine similarity are combined. "
            "This is not an official ATS score."
        ),
    }
