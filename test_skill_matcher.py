
from src.matching.skill_matcher import compare_skills

resume = """
Machine learning engineer with Python, PyTorch, pandas,
scikit-learn, NLP and FastAPI experience.
Built computer vision projects using YOLO.
"""

job_description = """
We need a candidate skilled in Python, machine learning,
deep learning, NLP, PyTorch, TensorFlow, SQL and Docker.
Experience with REST APIs is preferred.
"""

result = compare_skills(resume, job_description)

print("Resume skills:", result["resume_skills"])
print("Required skills detected:", result["job_skills"])
print("Matched skills:", result["matched_skills"])
print("Missing skills:", result["missing_skills"])
print("Keyword coverage:", result["keyword_score"], "%")
