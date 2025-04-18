import os
import re
import sys
import json
import base64
import subprocess
import zipfile
import datetime
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any, List
import logging
import networkx as nx 
from matplotlib import pyplot as plt
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from docx import Document
from langgraph.graph import StateGraph, END  # type: ignore[import]
from langchain_core.runnables import RunnableLambda  # type: ignore[import]
from langchain_core.messages import HumanMessage  # type: ignore[import]
from langsmith import Client
from utils.llms import groq_llm, gemini_llm, groq_llm_vision  # type: ignore[import]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "project_output_" + current_datetime
os.makedirs(output_dir, exist_ok=True)

# Database setup for persistence
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://lguser:lgpass@localhost:5432/langgraph")
engine = sa.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class GraphPersistence(Base):
    __tablename__ = "graph_persistence"
    graph_id = sa.Column(sa.String, primary_key=True)
    state = sa.Column(JSONB)

Base.metadata.create_all(engine)

# LangSmith client\
langsmith = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Define state schema
class GraphState(TypedDict, total=False):
    file_path: str
    schema_image_path: Optional[str]
    srs_text: Optional[str]
    analysis: Optional[Dict[str, Any]]
    schema_from_image: Optional[Dict[str, Any]]
    setup_spec: Optional[Dict[str, Any]]
    output_dir: Optional[str]
    code_files: Dict[str, str]
    test_files: Dict[str, str]
    error_files: List[str]
    zip_path: Optional[str]
    documentation: Dict[str, str]
    langsmith_logs: List[Dict[str, Any]]

# Utility functions
def read_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def image_to_base64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

def run_pytest(test_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", str(test_dir)], capture_output=True, text=True)

# Persistence nodes
def init_persistence(state: GraphState) -> GraphState:
    graph_id = state.get("graph_id", "default")
    db = SessionLocal()
    rec = db.query(GraphPersistence).filter_by(graph_id=graph_id).first()
    if rec:
        state.update(rec.state)
    db.close()
    return state

def persist_state_node(state: GraphState) -> GraphState:
    graph_id = state.get("graph_id", "default")
    db = SessionLocal()
    rec = db.query(GraphPersistence).filter_by(graph_id=graph_id).first()
    if rec:
        rec.state = state
    else:
        rec = GraphPersistence(graph_id=graph_id, state=state)
        db.add(rec)
    db.commit()
    db.close()
    return state

# Milestone 1: Validate and read SRS
def validate_and_read_docx(state: GraphState) -> GraphState:
    path = Path(state.get("file_path", ""))
    if path.suffix.lower() != ".docx":
        raise ValueError("Invalid file format: expected .docx")
    text = read_docx(path)
    return {"srs_text": text}

# Milestone 1: Analyze SRS
def analyze_srs(state: GraphState) -> GraphState:
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
    return {"analysis": analysis}

# Convert image to base64 string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return None

# Analyze Schema Image using Vision-capable LLaMA model
def analyze_schema_image(state: GraphState) -> GraphState:
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
        return {"schema_from_image": extracted_schema}
    except Exception as e:
        return {"error": f"Vision model failed: {e}"}


# Milestone 2: Project setup
def project_setup(state: GraphState) -> GraphState:
    state["output_dir"] = output_dir
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
    return {**state, "setup_spec": spec}


def generate_tests(state: GraphState) -> GraphState:
    prompt = f"""
You are a senior FastAPI test engineer using Test-Driven Development (TDD) principles.

Follow this structure for the project:

project_root/
├── app/
│   ├── api/routes/
│   ├── models/
│   ├── services/
│   ├── main.py
│   └── database.py
├── tests/
├── requirements.txt

Based on this feature specification:

{json.dumps(state['analysis'], indent=2)}

Write Pytest test cases for all endpoints, services, and models. Cover normal + edge cases.

Return only JSON like:
{{
  "tests/test_user.py": "...",
  "tests/test_item.py": "..."
}}

Generate only JSON formatted response. No other text. No explanations.
"""
    resp = gemini_llm.invoke([HumanMessage(content=prompt)])
    content = re.sub(r'^```json|```$', '', resp.content.strip(), flags=re.MULTILINE)
    try:
        tests = json.loads(content)
    except:
        generate_tests(state)  
    state["test_files"] = tests
    logger.info("Test files generated.")
    return state

# Step 3: Generate code
def generate_code_with_gemini(state: GraphState) -> GraphState:
    prompt = f"""
You are a senior FastAPI backend engineer.

Project structure should follow:

project_root/
├── app/
│   ├── api/routes/
│   ├── models/
│   ├── services/
│   ├── main.py
│   └── database.py
├── requirements.txt
├── Dockerfile
├── .env
└── README.md

Based on:
Analysis: {json.dumps(state['analysis'], indent=2)}
Setup Spec: {json.dumps(state['setup_spec'], indent=2)}

Generate complete, modular, well-documented FastAPI code (routes, models, services, configs) with:
- Logging, error handling, docstrings
- Dummy database configs
- Ready-to-run structure

Return only JSON like:
{{
  "app/main.py": "...",
  "app/api/routes/user.py": "...",
  "app/models/user.py": "..."
}}

Generate only JSON formatted response. No other text. No explanations.
"""
    resp = gemini_llm.invoke([HumanMessage(content=prompt)])
    content = re.sub(r'^```json|```$', '', resp.content.strip(), flags=re.MULTILINE)
    try:
        code = json.loads(content)
    except:
        generate_code_with_gemini(state)
    state["code_files"] = code
    logger.info("Code files generated.")
    return state

# Step 3: Save code and tests
def save_code_and_tests(state: GraphState) -> GraphState:
    out = Path(state["output_dir"])
    for fp, txt in {**state["code_files"], **state["test_files"]}.items():
        p = out / fp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    logger.info("Code and tests saved.")
    return state

# Step 4: Run tests
def run_and_validate(state: GraphState) -> GraphState:
    test_dir = Path(state["output_dir"]) / "tests"
    res = run_pytest(test_dir)
    print(res)
    if res.returncode == 0:
        state["execution_success"] = True
        state["error_files"] = []
    else:
        state["execution_success"] = False
        state["error_files"] = re.findall(r'ERROR project_output/[\w_]+\.py', res.stdout)
        logger.error(state["error_files"])
    return state

def correct_code_with_llm(file_code):
    """
    Use the LLM to correct code errors.
    """
    prompt = f"""
    You are a senior FastAPI backend engineer. 
    You encountered an error in the code you generated.
    The code is:

    {file_code}

    Don't change any variable names or structure. Just rectify the error and provide only the corrected Python code — no other text or explanation.
    """
    
    try:
        response = gemini_llm.invoke([HumanMessage(content=prompt)])

        match = re.search(r'```python(.*?)```', response.content, re.DOTALL)
        return match.group(1) if match else response.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def refine_until_success(state):
    max_iterations = 3
    attempts = 0

    state = run_and_validate(state)

    while not state.get("execution_success") and attempts < max_iterations:
        print(f"Refinement attempt {attempts + 1}/{max_iterations}...")
        error_files = state.get("error_files", [])
        print(f"Error files in refine: {error_files}")

        for file in error_files:
            print("Correcting file:", file)
            if not os.path.exists(file):
                print(f"⚠️ File not found: {file}")
                continue

            with open(file, "r") as f:
                original_code = f.read()

            corrected_code = correct_code_with_llm(original_code)
            if corrected_code:
                with open(file, "w") as f:
                    f.write(corrected_code)

        # Re-run tests after corrections
        state = run_and_validate(state)
        attempts += 1

    if state.get("execution_success"):
        print("Iterative refinement complete. All tests passed.")
    else:
        print("Maximum refinement attempts reached. Some tests may still be failing.")

    return state

# Deployment: zip
def zip_project(state: GraphState) -> GraphState:
    output_dir = state["output_dir"]
    zip_name = "project_output" + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + ".zip"
    full_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(full_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file == zip_name:
                    continue
                path = os.path.join(root, file)
                zf.write(path, arcname=os.path.relpath(path, "."))
   
    # Now clean up everything except the zip file
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if os.path.basename(path) != zip_name:
                os.remove(path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)

    return {**state, "zip_path": zip_name}

# Documentation via LLM
def json_to_plain_text(json_obj, indent=0):
    plain_text = ""
    for key, value in json_obj.items():
        plain_text += ' ' * indent + f"{key}: "
        if isinstance(value, dict):
            plain_text += "\n" + json_to_plain_text(value, indent + 4)
        else:
            plain_text += f"{value}\n"
    return plain_text

def generate_documentation(state: GraphState) -> GraphState:
    resp = gemini_llm.invoke([HumanMessage(content=f"Generate detailed documentation for code: {state['code_files']} Generate only JSON formatted response. No other text. No explanations.")])
    content = re.sub(r'^```json|```$', '', resp.content.strip(), flags=re.MULTILINE)
    try:
        state['documentation'] = json.loads(content)
        # Convert JSON to plain text
        plain_text = json_to_plain_text(state['documentation'])
        # Save the plain text documentation to a txt file
        with open(f'{state['output_dir']}/documentation.txt', 'w') as file:
            file.write(plain_text)
    except:
        generate_documentation(state)
    return state

# LangSmith logging (manual)
def track_langsmith(state: GraphState) -> GraphState:
    try:
        client = Client()

        output_dir = state.get("output_dir", "generated_project")
        os.makedirs(output_dir, exist_ok=True)

        project_name = state.get("langsmith_project") or os.getenv("LANGSMITH_PROJECT")
        if not project_name:
            raise ValueError("LangSmith project name not found in state or environment.")

        start_time = datetime.datetime.now() - datetime.timedelta(days=1)
        end_time = datetime.datetime.now()

        runs = client.list_runs(
            project=project_name,
            status="completed",
            start_time=start_time,
            end_time=end_time,
            limit=5
        )

        saved_runs = []
        for run in runs:
            run_dict = run.to_dict()
            run_file_path = os.path.join(output_dir, f"run_{run.id}.json")
            with open(run_file_path, "w") as f:
                json.dump(run_dict, f, indent=4)
            saved_runs.append(run_file_path)

        logger.info(f"✅ Saved {len(saved_runs)} LangSmith run(s) to: {output_dir}")
        state["saved_langsmith_runs"] = saved_runs

    except Exception as e:
       
        # Log key parts of the state for debugging
        debug_info = {
            "output_dir": state.get("output_dir"),
            "langsmith_project": state.get("langsmith_project"),
            "other_keys": list(state.keys())
        }

        logger.error(f"State during exception: {json.dumps(debug_info, indent=2)}")

    return state



def visualize_graph(steps):
    G = nx.DiGraph()
    G.add_nodes_from(steps)
    edges = [(a, b) for a, b in zip(steps, steps[1:])]
    edges.append((steps[-1], 'END'))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.title('LangGraph Pipeline')
    plt.savefig(f"{output_dir}/langgraph.png")

# Main execution
def main():

    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Run LangGraph pipeline')
    parser.add_argument('--file', required=True, help='SRS Document path')
    parser.add_argument('--schema', help='Path to schema image (optional)')
    parser.add_argument('--output', default=f'{output_dir}', help='Output directory')
    parser.add_argument('--graph-id', default='default', help='Persistence graph ID')
    args = parser.parse_args()

    state: GraphState = {
        'file_path': args.file,
        'schema_image_path': args.schema,
        'output_dir': args.output,
        'graph_id': args.graph_id
    }

    steps = [
        'init_persistence', 'validate_and_read_docx', 'analyze_srs', 'analyze_schema_image',
        'project_setup', 'generate_tests', 'generate_code_with_gemini', 'save_code_and_tests', 'run_and_validate', 'refine_until_success',        'persist_state_node', 'zip_project',
        'generate_documentation', 'track_langsmith'
    ]

    graph = StateGraph(GraphState)
    for name in steps:
        graph.add_node(name, RunnableLambda(globals()[name]))
    graph.set_entry_point('init_persistence')
    for a, b in zip(steps, steps[1:]):
        graph.add_edge(a, b)
    graph.add_edge(steps[-1], END)

    backend = graph.compile()
    result = backend.invoke(state)
    with open(f'{state['output_dir']}/final_state.json', 'w') as f:
        json.dump(result, f, indent=4)

    visualize_graph(steps)
    print("Completed pipeline")


if __name__ == '__main__':
    main()
