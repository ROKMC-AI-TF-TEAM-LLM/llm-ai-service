import re
from pathlib import Path


def format_docs(docs) -> str:
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


def strip_sources_section(text: str) -> str:
    match = re.search(r'\n{1,2}\*\*출처\*\*', text)
    if match:
        return text[:match.start()].rstrip()
    return text.rstrip()
