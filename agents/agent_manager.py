
from agents.structure_agent import StructureAgent
from agents.novelty_agent import NoveltyAgent
from agents.methodology_agent import MethodologyAgent
from agents.literature_agent import LiteratureAgent
from agents.experiment_agent import ExperimentAgent
from agents.writing_agent import WritingAgent
from agents.citation_agent import CitationAgent


class AgentManager:
    """
    Manages the seven specialist agents used by AutoReview.

    Each agent receives:
    1. The full text of the research paper.
    2. Evidence retrieved specifically for that agent.

    Agents run sequentially to avoid concurrent API requests.
    """

    def __init__(self):
        self.agents = [
            StructureAgent(),
            NoveltyAgent(),
            MethodologyAgent(),
            LiteratureAgent(),
            ExperimentAgent(),
            WritingAgent(),
            CitationAgent(),
        ]

    def run_single_agent(
        self,
        agent,
        paper_text,
        retrieved_evidence,
    ):
        """
        Run one specialist agent.

        retrieved_evidence may be either:
        - A dictionary mapping agent names to evidence strings.
        - A single evidence string for backward compatibility.
        """

        print("\n" + "=" * 70)
        print(f"Running: {agent.name}")
        print("=" * 70)

        # Select evidence intended for this specific agent.
        if isinstance(retrieved_evidence, dict):
            agent_evidence = retrieved_evidence.get(
                agent.name,
                "No external reference evidence was retrieved.",
            )
        else:
            # Preserve compatibility with the original shared-evidence workflow.
            agent_evidence = retrieved_evidence

        try:
            result = agent.review(
                paper_text,
                agent_evidence,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    f"{agent.name}.review() must return a dictionary."
                )

            review = result.get("review")

            if review is None or not str(review).strip():
                raise ValueError(
                    f"{agent.name} returned an empty review."
                )

            print(review)

            return agent.name, review

        except Exception as exc:
            print(f"\nERROR while running {agent.name}:")
            print(f"{type(exc).__name__}: {exc}")

            return agent.name, None

    def run_all(
        self,
        paper_text,
        retrieved_evidence,
    ):
        """
        Run all specialist agents sequentially.

        Args:
            paper_text: Full extracted text of the submitted paper.
            retrieved_evidence: Either shared evidence text or a
                dictionary mapping each agent name to its evidence.

        Returns:
            Dictionary mapping successful agent names to review text.
        """

        results = {}

        print("\n" + "=" * 70)
        print("RUNNING SPECIALIST AGENTS")
        print("=" * 70)

        if isinstance(retrieved_evidence, dict):
            print("\nEvidence mode: Agent-specific retrieval")
        else:
            print("\nEvidence mode: Shared retrieval")

        print("Execution mode: Sequential")
        print("Workers: 1")

        for agent in self.agents:
            agent_name, review = self.run_single_agent(
                agent,
                paper_text,
                retrieved_evidence,
            )

            if review is not None:
                results[agent_name] = review

        print("\n" + "=" * 70)
        print("SPECIALIST AGENTS COMPLETED")
        print("=" * 70)

        print(
            f"\nSuccessful agents: "
            f"{len(results)} / {len(self.agents)}"
        )

        failed_agents = [
            agent.name
            for agent in self.agents
            if agent.name not in results
        ]

        if failed_agents:
            print("\nAgents that failed:")
            for name in failed_agents:
                print(f" - {name}")

        return results
