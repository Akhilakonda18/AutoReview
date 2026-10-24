
import re


class ScoreFusion:
    def __init__(self):
        self.scores = {}

    def extract_score(self, review):
        if not review:
            return None

        # Matches SCORE: 9/10, CITATION SCORE: 7/10,
        # NOVELTY SCORE: 7.5/10, etc.
        pattern = (
            r"^\s*(?:(?:STRUCTURE|NOVELTY|METHODOLOGY|"
            r"LITERATURE|EXPERIMENT|WRITING|CITATION)\s+)?"
            r"SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10\b"
        )

        match = re.search(
            pattern,
            review,
            re.IGNORECASE | re.MULTILINE
        )

        if not match:
            return None

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
            else:
                print(
                    f"WARNING: No valid score found for "
                    f"{agent_name}"
                )

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
        "Structure Agent": "SCORE: 9/10",
        "Novelty Agent": "NOVELTY SCORE: 7/10",
        "Methodology Agent": "METHODOLOGY SCORE: 8.5/10",
        "Literature Agent": "LITERATURE SCORE: 8.5/10",
        "Experiment Agent": "EXPERIMENT SCORE: 8.5/10",
        "Writing Agent": "WRITING SCORE: 8.5/10",
        "Citation Agent": "CITATION SCORE: 7/10"
    }

    fusion = ScoreFusion()
    overall = fusion.calculate_overall_score(test_results)

    print("=" * 50)
    print("SCORE FUSION TEST")
    print("=" * 50)

    for agent, score in fusion.scores.items():
        print(f"{agent}: {score}/10")

    print("-" * 50)
    print(f"Successful agents: {len(fusion.scores)}/7")
    print(f"Overall Score: {overall}/10")
