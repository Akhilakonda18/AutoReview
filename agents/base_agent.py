class BaseAgent:

    def __init__(self, name):

        self.name = name

    def build_prompt(
        self,
        paper_text,
        retrieved_evidence
    ):

        raise NotImplementedError

    def review(
        self,
        paper_text,
        retrieved_evidence
    ):

        raise NotImplementedError