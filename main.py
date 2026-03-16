from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import ast
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GreenTrace AI Engine",
    description="Advanced AST-based code sustainability auditor",
    version="2.0.0"
)

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# --- Pydantic Models for API Requests/Responses ---
class CodeRequest(BaseModel):
    code: str

class MetricsResponse(BaseModel):
    lines_of_code: int
    functions: int
    classes: int
    loops: int
    conditions: int
    complexity_score: float
    estimated_energy_kwh: float
    carbon_footprint_grams: float
    green_grade: str

class FullAuditResponse(BaseModel):
    metrics: MetricsResponse
    ai_suggestions: str

# --- AST Analyzer Logic ---
class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stats = {
            "functions": 0,
            "classes": 0,
            "loops": 0,
            "conditions": 0,
        }

    def visit_FunctionDef(self, node):
        self.stats["functions"] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.stats["classes"] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.stats["loops"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.stats["loops"] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.stats["conditions"] += 1
        self.generic_visit(node)

def analyze_ast_metrics(code: str) -> MetricsResponse:
    try:
        # Parse the code into an Abstract Syntax Tree
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code: {e}")

    visitor = CodeVisitor()
    visitor.visit(tree)
    
    lines = len(code.strip().split('\n'))
    stats = visitor.stats
    
    # Advanced Heuristic Calculation
    complexity = (lines * 0.2) + (stats["loops"] * 8) + (stats["conditions"] * 3)
    complexity = max(1.0, complexity)
    
    energy_kwh = complexity * 0.003
    co2_grams = energy_kwh * 400 # 400g CO2 per kWh global avg
    
    # Determine Grade
    if complexity < 15:
        grade = "A"
    elif complexity < 35:
        grade = "B"
    elif complexity < 70:
        grade = "C"
    elif complexity < 120:
        grade = "D"
    else:
        grade = "E"

    return MetricsResponse(
        lines_of_code=lines,
        functions=stats["functions"],
        classes=stats["classes"],
        loops=stats["loops"],
        conditions=stats["conditions"],
        complexity_score=round(complexity, 2),
        estimated_energy_kwh=round(energy_kwh, 4),
        carbon_footprint_grams=round(co2_grams, 2),
        green_grade=grade
    )

# --- AI Integration ---
def get_ai_refactoring(code: str) -> str:
    if not API_KEY:
        return "API Key missing. Cannot generate AI suggestions."
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are an expert Sustainability Engineer. Analyze this Python code for energy and time complexity.
    Provide:
    1. Structural bottlenecks (CPU/Memory waste).
    2. A rewritten, greener version of the code.
    3. The expected impact on compute resources.
    
    Code:
    ```python\n{code}\n```
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error from AI model: {str(e)}"

# --- API Endpoints ---
@app.post("/api/v1/audit", response_model=FullAuditResponse)
async def audit_code(request: CodeRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")
        
    try:
        # 1. Run structural AST analysis
        metrics = analyze_ast_metrics(request.code)
        
        # 2. Get AI suggestions
        ai_response = get_ai_refactoring(request.code)
        
        return FullAuditResponse(
            metrics=metrics,
            ai_suggestions=ai_response
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
def root():
    return {"message": "GreenTrace AI API is running. Send a POST request to /api/v1/audit."}