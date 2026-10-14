
import re


def normalize_text(text: str) -> str:
    """Clean extracted text without changing its meaning."""

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.replace("\x00", " ")

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
