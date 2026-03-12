"""Parse the minimum needed from reviewer submission: verdict to control the loop."""


def parse_verdict(submission: str) -> dict:
    """Parse the reviewer's submission (contents of /tmp/review.txt).

    Returns {"raw": full text, "approved": bool}.
    """
    return {
        "raw": submission.strip(),
        "approved": "VERDICT: APPROVE" in submission,
    }
