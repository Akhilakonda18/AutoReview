import re


class ScoreFusion:

    def __init__(self):
        self.scores = {}

    def extract_score(self, review):

        if not review:
            return None

        patterns = [
            r"SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"NOVELTY SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"METHODOLOGY SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"LITERATURE SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"EXPERIMENT SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"WRITING SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"CITATION SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                review,
                re.IGNORECASE
            )

            if match:

                try:
                    score = float(match.group(1))

                    if 0 <= score <= 10:
                        return score

                except ValueError:
                    pass

        return None

    def calculate_overall_score(self, results):

        scores = {}

        for agent_name, review in results.items():

            score = self.extract_score(review)

            if score is not None:
                scores[agent_name] = score

        self.scores = scores

        if not scores:
            return 0.0

        overall = sum(scores.values()) / len(scores)

        return round(overall, 2)

    def get_summary(self):

        return {
            "scores": self.scores,
            "successful_agents": len(self.scores)
        }


if __name__ == "__main__":

    test_results = {

        "Structure Agent":
            "SCORE: 9/10",

        "Novelty Agent":
            "NOVELTY SCORE: 7/10",

        "Methodology Agent":
            "METHODOLOGY SCORE: 8/10",

        "Literature Agent":
            "LITERATURE SCORE: 7/10",

        "Experiment Agent":
            "EXPERIMENT SCORE: 7/10",

        "Writing Agent":
            "WRITING SCORE: 8/10",

        "Citation Agent":
            "CITATION SCORE: 7/10"
    }

    fusion = ScoreFusion()

    overall = fusion.calculate_overall_score(
        test_results
    )

    print("=" * 60)
    print("SCORE FUSION TEST")
    print("=" * 60)

    for agent, score in fusion.scores.items():

        print(
            f"{agent}: {score}/10"
        )

    print(
        "\nOverall Score:",
        overall,
        "/10"
    )