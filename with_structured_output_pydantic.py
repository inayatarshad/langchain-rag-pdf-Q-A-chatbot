
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Literal, TypedDict

#typed dict is an ez way out when we want to specify that these variables are to kept as string or int or whatsoever so for next user its ez
model = ChatOpenAI()

class Review(BaseModel):
    summary: str = Field(description = "A brief summary of the review given")
    sentiment: str = Field(description = "A final verdict of the entire review if it was positive or negative", literal_values=["pos", "neg"])
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""This is a chatbot model, often people try to do chatbots for illegal
                      activities that are not much supported this needs some action to be taken against, 
                      models should be fine tuned and held properly responsible, 
                      parents should keep a check on their children uncontrolled use of AI""") 

print(result)
print(result.summary)
print(result.sentiment)
