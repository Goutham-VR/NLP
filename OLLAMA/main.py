from ollama import chat

# Send a message to the locally running LLM
response = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Explain NLP in simple terms."
        }
    ]
)

# Display the generated response
print(response.message.content)
