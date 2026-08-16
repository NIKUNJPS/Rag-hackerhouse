"""
First-pass filter for obviously unsafe/inappropriate queries before they
ever reach retrieval or generation. Deliberately simple (keyword screen) --
for a hackathon demo, transparent and explainable beats a black-box
classifier that's hard to justify to judges in a 90s video.
"""

import re

UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build|synthesize)\b.*\b(bomb|explosive|weapon|virus|malware)\b",
    r"\bhack\b.*\b(account|password|system|network)\b",
    r"\bself[\s-]?harm\b",
    r"\bkill (myself|yourself|someone)\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def is_unsafe(query: str) -> bool:
    return any(p.search(query) for p in _compiled)
