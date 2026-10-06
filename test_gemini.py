from core.llm_service import ask_qwen

response = ask_qwen(
    "Explain Retrieval Augmented Generation in 3 simple sentences."
)

print("\nGEMINI RESPONSE:")
print(response)