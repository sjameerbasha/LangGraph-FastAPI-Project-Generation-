from langchain_groq import ChatGroq # type: ignore[import]
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
import os
from docx import Document # type: ignore[import]

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

#read the file, validate and convert it to text
def validate_and_read_doc(file_path):
    if not file_path.lower().endswith(".docx"):
        return "Invalid file format. Please upload a .docx file."
    try:
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        print("Validation Done")
        return text
    except Exception as e:
        return f"Failed to read the document: {e}"
    
# check if the file is valid and read it   
print("Doc reader check: ", validate_and_read_doc("Python Gen AI SRD backend 14th 18th Apr (1).docx"))