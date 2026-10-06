import os
from datetime import datetime


def generate_report(
    paper_name,
    agent_results,
    overall_score,
    final_review
):

    output_dir = "outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    report_path = os.path.join(
        output_dir,
        "autoreview_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "AUTOREVIEW - RESEARCH PAPER REVIEW REPORT\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            f"Paper: {paper_name}\n"
        )

        file.write(
            f"Generated: {timestamp}\n\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "INDIVIDUAL AGENT RESULTS\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        for agent_name, result in (
            agent_results.items()
        ):

            file.write(
                "-" * 80 + "\n"
            )

            file.write(
                f"{agent_name}\n"
            )

            file.write(
                "-" * 80 + "\n"
            )

            file.write(
                result
            )

            file.write(
                "\n\n"
            )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "SCORE FUSION\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            f"Overall Score: "
            f"{overall_score:.2f}/10\n\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "FINAL CONSOLIDATED REVIEW\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            final_review
        )

        file.write(
            "\n\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            "END OF REPORT\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

    return report_path