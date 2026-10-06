import re


# Weights for the final AutoReview score
WEIGHTS = {
    "Structure Agent": 0.10,
    "Novelty Agent": 0.20,
    "Methodology Agent": 0.20,
    "Literature Agent": 0.10,
    "Experiment Agent": 0.15,
    "Writing Agent": 0.10,
    "Citation Agent": 0.05,
    "Final Reviewer Agent": 0.10,
}


def extract_score(text):
    """
    Extract the first score written as X/10.
    """

    if not text:
        return None

    match = re.search(
        r"(?:SCORE|score)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        text
    )

    if match:
        score = float(match.group(1))

        # Keep score between 0 and 10
        return max(0.0, min(10.0, score))

    return None


def calculate_final_score(agent_results):
    """
    Calculate weighted final score from agent results.
    """

    total = 0.0
    used_weight = 0.0

    scores = {}

    for agent_name, weight in WEIGHTS.items():

        result = agent_results.get(agent_name, "")

        score = extract_score(result)

        scores[agent_name] = score

        if score is not None:

            total += score * weight
            used_weight += weight

    if used_weight == 0:
        return None, scores

    # Normalize if some agents did not provide scores
    final_score = total / used_weight

    return round(final_score, 2), scores


def get_rating(score):

    if score is None:
        return "Unable to calculate"

    if score >= 9:
        return "Excellent"

    elif score >= 8:
        return "Very Good"

    elif score >= 7:
        return "Good"

    elif score >= 6:
        return "Needs Improvement"

    elif score >= 5:
        return "Weak"

    else:
        return "Poor"


if __name__ == "__main__":

    # Test data only
    test_results = {

        "Structure Agent":
            "SCORE: 8/10",

        "Novelty Agent":
            "NOVELTY SCORE: 6/10",

        "Methodology Agent":
            "METHODOLOGY SCORE: 7/10",

        "Literature Agent":
            "LITERATURE SCORE: 8/10",

        "Experiment Agent":
            "EXPERIMENT SCORE: 6/10",

        "Writing Agent":
            "WRITING SCORE: 9/10",

        "Citation Agent":
            "CITATION SCORE: 7/10",

        "Final Reviewer Agent":
            "OVERALL SCORE: 7/10",
    }

    final_score, scores = calculate_final_score(
        test_results
    )

    print("=" * 60)
    print("AUTOREVIEW SCORE ENGINE")
    print("=" * 60)

    print("\nIndividual Scores:")

    for agent, score in scores.items():
        print(f"{agent}: {score}/10")

    print("\nFinal Score:", final_score, "/10")

    print(
        "Rating:",
        get_rating(final_score)
    )