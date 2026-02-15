from langchain_anthropic import ChatAnthropic, ChatAnthropiv
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2", temperature=0.7, max_tokens=512)

result=model.invoke("what is capital of pakistan?")
print(result.content)