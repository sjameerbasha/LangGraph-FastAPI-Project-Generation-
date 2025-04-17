from langgraph.graph import StateGraph, END # type: ignore[import]
from langchain_core.runnables import RunnableLambda # type: ignore[import]
from langchain_core.messages import HumanMessage # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
from docx import Document
import os, re, json, base64
from typing import TypedDict, Optional, Dict, Any, List
import networkx as nx
import matplotlib.pyplot as plt
from utils.llms import groq_llm, gemini_llm, groq_llm_vision  # type: ignore[import]

# Load environment variables from .env file
load_dotenv()

# Validate & Read .docx File
def validate_and_read_docx(state):
    file_path = state["file_path"]
    if not file_path.lower().endswith(".docx"):
        return {"error": "Invalid file format. Please upload a .docx file."}
    try:
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        print('SRS Document Validation Done')
        return {"srs_text": text}
    except Exception as e:
        return {"error": f"Failed to read document: {e}"}

# Analyzing SRS Text
def analyze_srs(state):
    text = state["srs_text"]
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
        Return the JSON only. No other text. No explanations.
    """
    resp = groq_llm.invoke(prompt)
    match = re.match(r"```json\n(.*?)\n```", resp.content, re.DOTALL)
    content = match.group(1) if match else resp.content
    try:
        analysis = json.loads(content)
    except Exception as e:
        return {"error": f"Failed to parse analysis: {e}"}
    print('SRS Analysis Done')
    return {"analysis": analysis}

# Convert image to base64 string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return None

# Analyze Schema Image using Vision-capable LLaMA model
def analyze_schema_image(state):
    image_path = state.get("schema_image_path")
    if not image_path:
        return {"error": "No image provided."}

    image_b64 = image_to_base64(image_path)
    if not image_b64:
        return {"error": "Failed to read or encode image."}

    # Vision prompt
    vision_prompt = """
        You are an expert software architect.

        You will be shown an image of a database schema (ER diagram or table format).
        Extract the database tables, fields, data types if available, and relationships in structured JSON format.
        Return only the JSON output. Do not include any explanation or extra text.
    """

    try:
        resp = groq_llm_vision.invoke(vision_prompt, image=image_b64)
        match = re.match(r"```json\n(.*?)\n```", resp.content, re.DOTALL)
        content = match.group(1) if match else resp.content
        extracted_schema = json.loads(content)
        print('Schema Image Analysis Done')
        return {"schema_from_image": extracted_schema}
    except Exception as e:
        return {"error": f"Vision model failed: {e}"}

# Milestone 2: Project Setup
def project_setup(state):
    output_dir = state.get("output_dir", "generated_project")
    os.makedirs(output_dir, exist_ok=True)

    prompt = f"""
        Using this analysis:

        {json.dumps(state["analysis"], indent=2)}

        Create a JSON response defining the FastAPI project folder structure like:
        - app/api/routes/*.py
        - app/models/*.py
        - services/, database.py, main.py, README.md, .env, etc.

        Also return:
        - dependencies: ["fastapi", "uvicorn", ...]
        - initial_files: {{ "README.md": "...", ".env": "...", "Dockerfile": "...", ... }}
        - README.md should include all the project overview, explanation, endpoints and their descriptions, project structure, and how to run the project.
        - .env should include all the environment variables needed for the project.
        - Dockerfile should include all the dependencies needed to run the project.
        - The project should be ready to run with all the dependencies and files needed.

        Just provide the structure in JSON information without any extra information.
    """
    resp = groq_llm.invoke([HumanMessage(content=prompt)])
    match = re.match(r"```json\n(.*?)\n```", resp.content, re.DOTALL)
    analysis = match.group(1) if match else resp.content
    spec = json.loads(analysis)

    # Create folders
    for folder in ["app/api/routes", "app/models", "app/services", "tests"]:
        folder_path = os.path.join(output_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

    # Write files
    for rel_path, content in spec["initial_files"].items():
        full_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Write requirements.txt
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write("\n".join(spec["dependencies"]))
    print('Project Setup Done')
    return {**state, "setup_spec": spec}

# Define the state schema
class GraphState(TypedDict, total=False):
    file_path: str
    schema_image_path: Optional[str]
    srs_text: Optional[str]
    analysis: Optional[Dict[str, Any]]
    schema_from_image: Optional[Dict[str, Any]]
    error: Optional[str]
    setup_spec: Optional[Dict[str, Any]]
    output_dir: Optional[str]

# Build LangGraph
graph = StateGraph(GraphState) 

# Add nodes
graph.add_node("validate_and_read_docx", RunnableLambda(validate_and_read_docx))
graph.add_node("analyze_srs", RunnableLambda(analyze_srs))
graph.add_node("analyze_schema_image", RunnableLambda(analyze_schema_image))
graph.add_node("project_setup", RunnableLambda(project_setup))

# Define flow
graph.set_entry_point("validate_and_read_docx")
graph.add_edge("validate_and_read_docx", "analyze_srs")
graph.add_edge("analyze_srs", "analyze_schema_image")
graph.add_edge("analyze_schema_image", "project_setup")
graph.add_edge("project_setup", END)

# Compile the graph
backend_graph = graph.compile()

# Execute the Graph
result = backend_graph.invoke({
    "file_path": "Python Gen AI SRD backend 14th 18th Apr (1).docx",
    "schema_image_path": "srs_db_schema_screenshot.png",
    "output_dir": os.getenv("OUTPUT_DIR")
})

# Build NetworkX graph
G = nx.DiGraph()
G.add_edge("validate_and_read_docx", "analyze_srs")
G.add_edge("analyze_srs", "analyze_schema_image")
G.add_edge("analyze_schema_image", "project_setup")
G.add_edge("project_setup", "END")

# Plot and save
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(
    G, pos,
    with_labels=True,
    arrows=True,
    node_color="skyblue",
    node_size=2500,
    font_size=10
)
plt.title("LangGraph Flow")
plt.tight_layout()
plt.savefig("milestone1-milestone2-langgraph.png")  # Save the image

print("Graph saved as 'milestone1-milestone2-langgraph.png'")

# Output Result
print(json.dumps(result, indent=2))