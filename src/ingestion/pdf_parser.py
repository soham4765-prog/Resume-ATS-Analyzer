
import pymupdf


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract selectable text from PDF bytes."""

    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        ) as document:
            pages = [
                page.get_text("text")
                for page in document
            ]

    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(
            "The PDF could not be read."
        ) from exc

    text = "\n".join(pages).strip()

    if not text:
        raise ValueError(
            "No selectable text found. "
            "The PDF may be scanned or image-based."
        )

    return text
