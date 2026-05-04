from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "mistral-small",
    model_provider="mistralai",
    temperature=0.9,
    max_tokens=20,
    
)

response = model.invoke("write a poem on ai")
print(response.content)