import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    """Load the cached embedding model once and reuse it."""
    try:
        return SentenceTransformer(
            MODEL_NAME,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the cached semantic model. "
            "Verify that all model files are available locally."
        ) from exc


def split_into_chunks(text: str) -> list[str]:
    """Split text into reasonably short sentence-level chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = re.split(r"(?<=[.!?])\s+", text)

    return [
        chunk.strip()
        for chunk in chunks
        if len(chunk.strip()) >= 20
    ]


def compare_semantically(
    resume_text: str,
    job_description: str,
) -> dict:
    """
    Compare each job-description sentence with the most
    semantically similar resume sentence.

    Returns cosine similarities, not hiring probabilities.
    """
    resume_chunks = split_into_chunks(resume_text)
    job_chunks = split_into_chunks(job_description)

    if not resume_chunks:
        raise ValueError(
            "Not enough resume text for semantic matching."
        )

    if not job_chunks:
        raise ValueError(
            "Not enough job-description text for semantic matching."
        )

    model = load_embedding_model()

    resume_embeddings = model.encode(
        resume_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    job_embeddings = model.encode(
        job_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Normalized vectors: dot product equals cosine similarity.
    similarity_matrix = np.matmul(
        job_embeddings,
        resume_embeddings.T,
    )

    results = []

    for index, job_chunk in enumerate(job_chunks):
        best_index = int(np.argmax(similarity_matrix[index]))
        best_similarity = float(similarity_matrix[index, best_index])

        results.append({
            "requirement": job_chunk,
            "resume_evidence": resume_chunks[best_index],
            "similarity": best_similarity,
        })

    average_similarity = float(
        np.mean([item["similarity"] for item in results])
    )

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return {
        "average_similarity": average_similarity,
        "comparisons": results,
        "requirements_compared": len(job_chunks),
    }