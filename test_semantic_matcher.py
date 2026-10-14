
from src.semantic.semantic_matcher import compare_semantically


resume = """
Built backend services using FastAPI and Python.
Created REST endpoints and integrated external services.
Developed computer vision applications using YOLO.
Worked with pandas and scikit-learn for data analysis.
"""

job_description = """
The candidate should have experience developing REST APIs.
Experience with Python backend development is required.
Knowledge of deep learning and object detection is desirable.
The candidate should have experience with cloud infrastructure.
"""

result = compare_semantically(resume, job_description)

print(
    "Average cosine similarity:",
    round(result["average_similarity"], 3),
)

print("\nRequirement-level comparisons:")

for item in result["comparisons"]:
    print(f"\nSimilarity: {item['similarity']:.3f}")
    print("Job requirement:", item["requirement"])
    print("Closest resume evidence:", item["resume_evidence"])
