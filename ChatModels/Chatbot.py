from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 

model = init_chat_model(
    "mistral-small",
    model_provider="mistralai",
    temperature=0.9,
)

print("choose your Ai mode")
print("1. Funny AI")
print("2. angry AI")
print("3. sad AI")
choice = input("Enter your choice: ")

if choice == "1":
    mode="You are a funny ai agent"
    
elif choice == "2":
    mode="You are an angry ai agent"
elif choice == "3":
    mode="You are a sad ai agent"

messages = [SystemMessage(content=mode)]


print("------------------Welcome type 0 to exit the application------------------")
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("AI: " + response.content)

print(messages)