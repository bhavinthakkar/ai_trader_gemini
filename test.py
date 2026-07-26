from ollama import chat

response = chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain RSI in one paragraph."
        }
    ]
)

print(response["message"]["content"])