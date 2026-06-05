import random
from app.config import (
    HIGH_AMOUNT,
    MED_AMOUNT,
    LOW_AMOUNT,
    SCORE_NOISE_MIN,
    SCORE_NOISE_MAX,
)


def score_transaction(amount: float, transaction_type: str) -> float:
    score = 0.0

    if amount >= HIGH_AMOUNT:
        score += 0.65
    elif amount >= MED_AMOUNT:
        score += 0.40
    elif amount >= LOW_AMOUNT:
        score += 0.20
    else:
        score += 0.05

    if transaction_type == "TRANSFER":
        score += 0.15
    elif transaction_type == "WITHDRAWAL":
        score += 0.05

    score += random.uniform(SCORE_NOISE_MIN, SCORE_NOISE_MAX)

    score = max(0.0, min(1.0, score))  # To keep the score between 0.0 and 1.0

    return round(score, 2)
