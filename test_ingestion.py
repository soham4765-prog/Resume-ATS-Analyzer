
from pathlib import Path

from src.ingestion.pdf_parser import extract_pdf_text
from src.ingestion.docx_parser import extract_docx_text
from src.preprocessing.clean_text import normalize_text



from pathlib import Path

raw_path = r"C:\Users\U S E R\Downloads\Sample_ML_Engineer_Intern_Resume.docx"
raw_path = raw_path.strip('"').strip("'")

resume_path = Path(raw_path)

print("DEBUG - Entered path:", repr(raw_path))
print("DEBUG - Resolved path:", resume_path.resolve())
print("DEBUG - Exists:", resume_path.exists())
print("DEBUG - Is file:", resume_path.is_file())
print("DEBUG - Is directory:", resume_path.is_dir())

resume_path = Path(raw_path)


if not resume_path.exists():
    print("ERROR: File or folder does not exist.")
    raise SystemExit(1)

if not resume_path.is_file():
    print("ERROR: The supplied path points to a folder, not a file.")
    raise SystemExit(1)

file_bytes = resume_path.read_bytes()
suffix = resume_path.suffix.lower()

try:
    if suffix == ".pdf":
        extracted = extract_pdf_text(file_bytes)
    elif suffix == ".docx":
        extracted = extract_docx_text(file_bytes)
    else:
        raise ValueError("Only PDF and DOCX files are supported.")

    cleaned = normalize_text(extracted)

    print("\n--- EXTRACTION SUCCESSFUL ---")
    print("File:", resume_path.name)
    print("Characters extracted:", len(cleaned))
    print("\n--- TEXT PREVIEW ---")
    print(cleaned[:2000])

except (ValueError, OSError) as exc:
    print("EXTRACTION FAILED:", exc)
