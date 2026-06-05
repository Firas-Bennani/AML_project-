# app/config.py
import os

# Risk Scoring
# FLAG_THRESHOLD: a transaction whose risk_score is >= this value is FLAGGED.
# Configurable via the BACKEND_FLAG_THRESHOLD env var so ops can tune without
# a code change. RISK_THRESHOLD is kept as an alias for backward compatibility.
FLAG_THRESHOLD = float(os.getenv("BACKEND_FLAG_THRESHOLD", "0.5"))
RISK_THRESHOLD = FLAG_THRESHOLD
HIGH_AMOUNT = 50000  # Score contribution: +0.65
MED_AMOUNT = 10000  # Score contribution: +0.40
LOW_AMOUNT = 5000  # Score contribution: +0.20

SCORE_NOISE_MIN = -0.05  # Lower bound of random noise added to each score
SCORE_NOISE_MAX = 0.10  # Upper bound of random noise added to each score

# Alert Severity Thresholds
SEVERITY_CRITICAL = 0.85  # risk_score >= this → CRITICAL
SEVERITY_HIGH = 0.75  # risk_score >= this → HIGH
# anything below HIGH → MEDIUM

# Analyst Performance Red Flag
RED_FLAG_DISMISSAL_RATE = 80  # % dismissal rate that triggers red flag
RED_FLAG_MIN_TRANSACTIONS = 5  # minimum reviews before red flag can apply

# Security
MAX_LOGIN_ATTEMPTS = 5  # account locks after this many failures
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # JWT token lifetime in minutes
ALGORITHM = "HS256"

# Pagination
DEFAULT_PAGE_SIZE = 20  # default number of items returned per page
DEFAULT_PAGE_START = 0  # default offset (start from the beginning)

# Reports
ALL_TIME_START = "2000-01-01"  # used as "beginning of time" for all-time reports
