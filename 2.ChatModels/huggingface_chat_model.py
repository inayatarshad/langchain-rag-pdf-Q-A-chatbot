from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Change task to "conversational" if "text-generation" fails for Chat models
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation", # or try removing this line to let HF auto-detect
    max_new_tokens=512,
    temperature=0.7
    )

model= ChatHuggingFace(llm=llm)
result=model.invoke("what is the capital of india?")
print(result.content)
