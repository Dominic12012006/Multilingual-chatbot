from asr import translate_audio
from llm import chat
from translate import translation
from tts import speak
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

while True:
    choice=input("Enter 1 to talk: ,2 to exit: ")
    if choice=='1':
        user_input=translate_audio("hindi","translate")
        print("You: ",user_input)
        chat_output=chat(user_input)
        response=translation("en_XX","hi_IN",chat_output)
        print("Bot: ",response)
        speak(response)
       
    elif choice=='2':
        break
    else:
        continue