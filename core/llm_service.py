import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL = "gemini-3.5-flash-lite"


def ask_qwen(prompt, max_tokens=1200):

    max_retries = 3

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"\nGemini request "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.2
                }
            )

            return response.text

        except Exception as e:

            error_text = str(e)

            print("\nGEMINI API ERROR:")
            print(error_text)

            # -----------------------------
            # DAILY QUOTA EXCEEDED
            # -----------------------------
            if "GenerateRequestsPerDay" in error_text:
                raise RuntimeError(
                    "Gemini daily quota exceeded. "
                    "Please wait for the quota reset "
                    "or use a project with available quota."
                )

            # -----------------------------
            # TEMPORARY RATE LIMIT
            # -----------------------------
            if "429" in error_text:

                if attempt < max_retries:
                    wait_time = 30 * attempt

                    print(
                        f"Rate limit reached. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

                raise RuntimeError(
                    "Gemini rate limit exceeded."
                )

            # -----------------------------
            # TEMPORARY SERVER ERROR
            # -----------------------------
            if "503" in error_text:

                if attempt < max_retries:

                    wait_time = 10 * attempt

                    print(
                        "Gemini temporarily unavailable."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Gemini model is temporarily "
                    "unavailable after multiple retries."
                )

            # -----------------------------
            # OTHER ERROR
            # -----------------------------
            raise RuntimeError(
                f"Gemini API Error: {e}"
            )

    raise RuntimeError(
        "Gemini request failed after retries."
    )