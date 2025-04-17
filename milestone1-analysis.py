from langchain_groq import ChatGroq # type: ignore[import]
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
import os

# Load environment variables from .env file
load_dotenv() # type: ignore[import]

# Initialize the Groq LLM
groq_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = os.getenv("GROQ_API_KEY")
)

# Initialize the Google GenAI LLM
gemini_llm = ChatGoogleGenerativeAI(
    temperature=0,
    model = "gemini-2.0-flash",
    max_tokens=None,
    max_retries=2,
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize Groq LLM with Llama 3 70B model
groq_llm_70b = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = os.getenv("GROQ_API_KEY")
)

# Check if the LLMs are initialized correctly
print("Groq LLM with llama-4: ", groq_llm.invoke("Hello, how are you?").content)
print("Google GenAI LLM: ", gemini_llm.invoke("Hello, how are you?").content)
print("Groq LLM with llama-3.3-70b: ", groq_llm_70b.invoke("Hello, how are you?").content)