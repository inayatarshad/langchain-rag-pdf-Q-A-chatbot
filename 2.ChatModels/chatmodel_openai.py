from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI(model="gpt-3.5-turbo")
result= model.invoke("what is the capital of india")
print(result)

