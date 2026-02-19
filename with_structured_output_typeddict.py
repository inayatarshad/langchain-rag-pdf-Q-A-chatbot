from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

#huggging face compatible ni hai structured output kay sath ic liye open ai use kr lo ,rn we dont hv api just for practice we use open ai in this file 
from langchain_openai import ChatOpenAI 
from transformers import pipeline
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI(model="gpt-4o")


class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str,"The sentiment of the review, either positive or negative"]

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""The hardware of the PC is nice yet there are some problems for the configuration of GPU stuff, special care is needed while tackling the issues related to the GPU. The CPU is also good but the gpu is not upto the mark. Hoping for an update to fix this.""")

print(result)
print(result["summary"])
print(result["sentiment"])