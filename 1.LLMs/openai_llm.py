from langchain_openai import OPENAI
from dotenv import load_dotenv

load_dotenv()

llm= OPENAI(model='gpt-3.5-turbo-instruct')

result=llm.invoke("what is the capital of india?")
print(result)
