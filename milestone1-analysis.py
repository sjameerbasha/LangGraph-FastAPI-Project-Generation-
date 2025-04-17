from langchain_groq import ChatGroq # type: ignore[import]
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
import os, re, json
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

# Analyze SRS text with Groq llm
def analyze_srs(text):
    prompt = f"""
Given this Software Requirements Spec (SRS):

{text}

Extract a JSON analysis like:
{{
  "endpoints": [{{"method": "GET", "path": "/users", "params": [], "auth_required": true}}],
  "logic": "...",
  "schema": [{{"table": "users", "fields": [...], "relationships": [...]}}],
  "auth": "..."
}}
Return the JSON only, no other text. No explanations.
"""
    resp = groq_llm.invoke(prompt)
    # convert the response to json
    match = re.match(r"```json\n(.*?)\n```", resp.content, re.DOTALL)
    analysis = match.group(1) if match else resp.content
    analysis = json.loads(analysis)
    print("Analysis Done")
    return analysis

# checking if the llm is able to analyze the srs text
text = validate_and_read_doc("Python Gen AI SRD backend 14th 18th Apr (1).docx")
print(analyze_srs(text))