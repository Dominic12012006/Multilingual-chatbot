from groq import Groq

client = Groq(
    api_key='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',
)
def chat(user):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content
#print(chat("namaste"))
