from datetime import datetime, timezone
from typing import Optional, Tuple
import re

TIME_WINDOWS = {
    "Q4_2024": (1727740800, 1735689599),
    "Q1_2025": (1735689600, 1743379199),
    "Q2_2025": (1743379200, 1750636799),
}


def epoch_to_window(created_utc: float) -> str:
    for label, (start, end) in TIME_WINDOWS.items():
        if start <= created_utc <= end:
            return label
    if created_utc > 1750636799:
        return "Q2_2025"
    return "Q4_2024"


def parse_time_expression(text: str) -> Optional[Tuple[float, float]]:
    text_lower = text.lower()
    now = datetime.now(timezone.utc).timestamp()

    patterns = [
        (r"last\s+(\d+)\s+month", lambda m: (now - int(m.group(1)) * 30 * 86400, now)),
        (r"past\s+(\d+)\s+month", lambda m: (now - int(m.group(1)) * 30 * 86400, now)),
        (r"last\s+6\s+month", lambda m: (now - 180 * 86400, now)),
        (r"last\s+year", lambda m: (now - 365 * 86400, now)),
        (r"last\s+month", lambda m: (now - 30 * 86400, now)),
        (r"last\s+week", lambda m: (now - 7 * 86400, now)),
        (r"q4\s*2024", lambda m: TIME_WINDOWS["Q4_2024"]),
        (r"q1\s*2025", lambda m: TIME_WINDOWS["Q1_2025"]),
        (r"q2\s*2025", lambda m: TIME_WINDOWS["Q2_2025"]),
        (r"oct(?:ober)?\s+2024", lambda m: TIME_WINDOWS["Q4_2024"]),
        (r"jan(?:uary)?\s+2025", lambda m: TIME_WINDOWS["Q1_2025"]),
    ]

    for pattern, handler in patterns:
        m = re.search(pattern, text_lower)
        if m:
            try:
                return handler(m)
            except Exception:
                continue
    return None


def parse_comparison_windows(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for label in TIME_WINDOWS:
        if label.lower().replace("_", " ") in text_lower or label.lower() in text_lower:
            found.append(label)
    return found if len(found) >= 2 else []
