from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repitition_penalty=1.03,
    )
)

chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("What is data Science ?")
print(response.content)