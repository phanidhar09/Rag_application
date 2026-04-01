from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/llama-3-70b-instruct",   
    base_url="https://openrouter.ai/api/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

response = llm.invoke("Who will win this years IPL")

print(response.content)
