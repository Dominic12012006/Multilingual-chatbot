from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template='''
Answer The Question Below

Here is the conversation history: {context}

Question:{question}

Answer:

'''
model=OllamaLLM(model="mistral")
prompt=ChatPromptTemplate.from_template(template)
chain=prompt|model
context=""
def chata(user):
    global context
    #print("Welcome to chatbot")
    #user=input("You: ")
    #print("You:",user)
    result=chain.invoke({"context":context,"question":user})
    #print("Bot: ",result)
    context+=f"\nUser:{user}\nAI:{result}"
    return result

