from agents.structure_agent import StructureAgent
from agents.novelty_agent import NoveltyAgent
from agents.methodology_agent import MethodologyAgent
from agents.literature_agent import LiteratureAgent
from agents.experiment_agent import ExperimentAgent
from agents.writing_agent import WritingAgent
from agents.citation_agent import CitationAgent


class AgentManager:

    def __init__(self):

        self.agents = [
            StructureAgent(),
            NoveltyAgent(),
            MethodologyAgent(),
            LiteratureAgent(),
            ExperimentAgent(),
            WritingAgent(),
            CitationAgent()
        ]

    def run_single_agent(
        self,
        agent,
        paper_text,
        retrieved_evidence
    ):

        print("\n" + "=" * 70)
        print(f"Running: {agent.name}")
        print("=" * 70)

        try:

            result = agent.review(
                paper_text,
                retrieved_evidence
            )

            review = result["review"]

            print(review)

            return agent.name, review

        except Exception as e:

            print(
                f"\nERROR while running {agent.name}:"
            )

            print(e)

            return agent.name, None

    def run_all(
        self,
        paper_text,
        retrieved_evidence
    ):

        results = {}

        print("\n")
        print("=" * 70)
        print("RUNNING SPECIALIST AGENTS")
        print("=" * 70)

        print(
            "\nExecution mode: Sequential"
        )

        print(
            "Workers: 1"
        )

        # -----------------------------------------------------
        # Run one agent at a time.
        # This avoids OpenRouter in-flight request problems.
        # -----------------------------------------------------

        for agent in self.agents:

            agent_name, review = self.run_single_agent(
                agent,
                paper_text,
                retrieved_evidence
            )

            if review is not None:

                results[agent_name] = review

        print("\n")
        print("=" * 70)
        print("SPECIALIST AGENTS COMPLETED")
        print("=" * 70)

        print(
            f"\nSuccessful agents: "
            f"{len(results)} / {len(self.agents)}"
        )

        return results