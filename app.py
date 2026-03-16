import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None

try:
    import nbformat
except Exception:
    nbformat = None


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="GreenTrace AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111a 0%, #0d1b2a 45%, #102a22 100%);
        color: #e6fff3;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7CFFB2, #8EE3FF, #C4FF5A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #bdebd3;
        margin-bottom: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(124,255,178,0.12), rgba(142,227,255,0.08));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #bdebd3;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
    }
    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e6fff3;
        margin: 0.8rem 0 0.6rem 0;
    }
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .good {
        background: rgba(62, 207, 142, 0.16);
        color: #8affc9;
        border: 1px solid rgba(62,207,142,0.4);
    }
    .medium {
        background: rgba(255, 193, 7, 0.14);
        color: #ffd76b;
        border: 1px solid rgba(255,193,7,0.35);
    }
    .bad {
        background: rgba(255, 99, 132, 0.15);
        color: #ff9ab1;
        border: 1px solid rgba(255,99,132,0.35);
    }
    .small-note {
        color: #a8cdb9;
        font-size: 0.88rem;
    }
    .footer-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px;
        color: #cce9d7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class AnalysisResult:
    file_name: str
    file_type: str
    lines_of_code: int
    num_functions: int
    num_classes: int
    num_loops: int
    nested_loops: int
    num_conditions: int
    num_comprehensions: int
    num_imports: int
    repeated_calls: int
    recursive_functions: int
    estimated_complexity_score: float
    energy_wh: float
    carbon_g_co2: float
    green_score: float
    rating: str
    suggestions: List[str]
    hotspots: List[Tuple[str, float]]


# -----------------------------
# Helpers: File Parsing
# -----------------------------
def extract_python_from_ipynb(raw_bytes: bytes) -> str:
    if nbformat is None:
        raise ValueError("nbformat is not installed. Run: pip install nbformat")

    try:
        nb = nbformat.reads(raw_bytes.decode("utf-8"), as_version=4)
    except Exception as e:
        raise ValueError(f"Could not read notebook file: {e}")

    code_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source.strip()
            if source:
                code_cells.append(source)

    if not code_cells:
        raise ValueError("No code cells found in the notebook.")

    return "\n\n".join(code_cells)


def extract_python_from_text(content: str, file_type: str) -> str:
    file_type = file_type.lower()

    if file_type in {"py", "python"}:
        return content

    if file_type in {"txt", "md", "markdown", "log", "csv", "json", "yaml", "yml", "ini", "toml"}:
        fenced_blocks = re.findall(
            r"```(?:python|py)?\n(.*?)```",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced_blocks:
            return "\n\n".join(block.strip() for block in fenced_blocks if block.strip())
        return content

    return content


def detect_file_type(file_name: str) -> str:
    if "." not in file_name:
        return "text"
    return file_name.rsplit(".", 1)[1].lower()


def load_code_from_uploaded_file(uploaded_file) -> Tuple[str, str, str]:
    file_name = uploaded_file.name
    file_type = detect_file_type(file_name)
    raw_bytes = uploaded_file.read()

    if file_type == "ipynb":
        code = extract_python_from_ipynb(raw_bytes)
        return code, file_name, file_type

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = raw_bytes.decode("latin-1")
        except Exception:
            raise ValueError("Unsupported encoding. Save the file as UTF-8 and try again.")

    code = extract_python_from_text(content, file_type)
    return code, file_name, file_type


# -----------------------------
# Static Analysis
# -----------------------------
def count_lines(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip()])


def detect_recursion(tree: ast.AST) -> int:
    recursive_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id == function_name:
                        recursive_count += 1
                        break
    return recursive_count


def get_loop_depth(node: ast.AST, current_depth: int = 0) -> int:
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            max_depth = max(max_depth, get_loop_depth(child, current_depth + 1))
        else:
            max_depth = max(max_depth, get_loop_depth(child, current_depth))
    return max_depth


def count_nested_loops(tree: ast.AST) -> int:
    nested = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            depth = get_loop_depth(node)
            if depth >= 2:
                nested += 1
    return nested


def detect_repeated_calls(tree: ast.AST) -> int:
    calls: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = None
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr

            if call_name:
                calls[call_name] = calls.get(call_name, 0) + 1

    return sum(1 for _, count in calls.items() if count >= 4)


def detect_hotspots(code: str) -> List[Tuple[str, float]]:
    hotspots = []
    for i, line in enumerate(code.splitlines(), start=1):
        score = 0.0
        stripped = line.strip()

        if re.search(r"\bfor\b|\bwhile\b", stripped):
            score += 2.0
        if ".append(" in stripped:
            score += 1.2
        if "range(len(" in stripped:
            score += 1.8
        if any(token in stripped for token in ["sort(", "sleep(", "requests.", "open(", "read_csv("]):
            score += 1.0
        if stripped.count("for ") >= 2:
            score += 2.5

        if score > 0:
            hotspots.append((f"Line {i}: {stripped[:80]}", score))

    hotspots.sort(key=lambda x: x[1], reverse=True)
    return hotspots[:10]


def estimate_complexity_score(metrics: Dict[str, int]) -> float:
    score = (
        metrics["num_loops"] * 2.0
        + metrics["nested_loops"] * 5.0
        + metrics["num_conditions"] * 1.2
        + metrics["num_functions"] * 0.8
        + metrics["repeated_calls"] * 2.5
        + metrics["recursive_functions"] * 3.0
        - metrics["num_comprehensions"] * 0.8
    )
    return max(1.0, round(score, 2))


def estimate_energy_and_carbon(
    metrics: Dict[str, int],
    carbon_intensity_g_per_kwh: float,
) -> Tuple[float, float]:
    energy_wh = (
        0.018 * metrics["lines_of_code"]
        + 0.65 * metrics["num_loops"]
        + 1.55 * metrics["nested_loops"]
        + 0.22 * metrics["num_conditions"]
        + 0.14 * metrics["num_functions"]
        + 0.10 * metrics["num_imports"]
        + 0.55 * metrics["repeated_calls"]
        + 0.90 * metrics["recursive_functions"]
        - 0.20 * metrics["num_comprehensions"]
    )
    energy_wh = max(0.2, round(energy_wh, 3))
    carbon_g_co2 = round((energy_wh / 1000) * carbon_intensity_g_per_kwh, 3)
    return energy_wh, carbon_g_co2


def compute_green_score(metrics: Dict[str, int], complexity_score: float, energy_wh: float) -> Tuple[float, str]:
    score = 100.0
    score -= metrics["nested_loops"] * 8
    score -= metrics["num_loops"] * 2.5
    score -= metrics["repeated_calls"] * 4
    score -= metrics["recursive_functions"] * 5
    score -= max(0, complexity_score - 8) * 1.8
    score -= max(0, energy_wh - 2.5) * 1.6
    score += min(metrics["num_comprehensions"] * 1.8, 6)

    score = max(10.0, min(98.0, round(score, 2)))

    if score >= 80:
        rating = "Excellent"
    elif score >= 60:
        rating = "Moderate"
    else:
        rating = "Needs Optimization"

    return score, rating


def generate_rule_based_suggestions(metrics: Dict[str, int], code: str) -> List[str]:
    suggestions = []

    if metrics["nested_loops"] > 0:
        suggestions.append("Reduce nested loops using dictionaries, sets, joins, or vectorized NumPy/Pandas operations.")
    if "range(len(" in code:
        suggestions.append("Replace range(len(...)) with direct iteration or enumerate() for cleaner and more efficient code.")
    if ".append(" in code and metrics["num_loops"] > 0:
        suggestions.append("Use list comprehensions where appropriate instead of repeated append() inside loops.")
    if metrics["repeated_calls"] > 0:
        suggestions.append("Cache repeated expensive function calls or store results to avoid recomputation.")
    if metrics["recursive_functions"] > 0:
        suggestions.append("Review recursion. Iterative approaches may be more memory-efficient for large inputs.")
    if "iterrows(" in code:
        suggestions.append("Avoid DataFrame.iterrows() on large datasets. Prefer vectorized pandas operations.")
    if "open(" in code and "with open(" not in code:
        suggestions.append("Use 'with open(...)' context managers to handle files efficiently and safely.")
    if any(imp in code for imp in ["tensorflow", "torch", "sklearn"]):
        suggestions.append("Load heavy ML libraries only when needed to reduce startup overhead and memory use.")
    if metrics["num_imports"] > 8:
        suggestions.append("Remove unused or heavy imports to cut startup overhead and improve maintainability.")

    if not suggestions:
        suggestions.append("The code looks reasonably efficient. Profile runtime hotspots to get deeper sustainability wins.")

    return suggestions[:8]


def analyze_code(code: str, carbon_intensity: float, file_name: str, file_type: str) -> AnalysisResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"Python parsing failed. Make sure the uploaded file contains valid Python code. Details: {e}"
        )

    metrics = {
        "lines_of_code": count_lines(code),
        "num_functions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
        "num_classes": sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
        "num_loops": sum(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for node in ast.walk(tree)),
        "nested_loops": count_nested_loops(tree),
        "num_conditions": sum(isinstance(node, ast.If) for node in ast.walk(tree)),
        "num_comprehensions": sum(
            isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
            for node in ast.walk(tree)
        ),
        "num_imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
        "repeated_calls": detect_repeated_calls(tree),
        "recursive_functions": detect_recursion(tree),
    }

    complexity_score = estimate_complexity_score(metrics)
    energy_wh, carbon_g_co2 = estimate_energy_and_carbon(metrics, carbon_intensity)
    green_score, rating = compute_green_score(metrics, complexity_score, energy_wh)
    suggestions = generate_rule_based_suggestions(metrics, code)
    hotspots = detect_hotspots(code)

    return AnalysisResult(
        file_name=file_name,
        file_type=file_type,
        lines_of_code=metrics["lines_of_code"],
        num_functions=metrics["num_functions"],
        num_classes=metrics["num_classes"],
        num_loops=metrics["num_loops"],
        nested_loops=metrics["nested_loops"],
        num_conditions=metrics["num_conditions"],
        num_comprehensions=metrics["num_comprehensions"],
        num_imports=metrics["num_imports"],
        repeated_calls=metrics["repeated_calls"],
        recursive_functions=metrics["recursive_functions"],
        estimated_complexity_score=complexity_score,
        energy_wh=energy_wh,
        carbon_g_co2=carbon_g_co2,
        green_score=green_score,
        rating=rating,
        suggestions=suggestions,
        hotspots=hotspots,
    )


# -----------------------------
# Gemini
# -----------------------------
def get_gemini_suggestions(code: str, api_key: str) -> str:
    if not api_key:
        return "Gemini API key not provided. Add it in the sidebar to enable AI suggestions."

    if genai is None:
        return "google-genai is not installed. Run: pip install google-genai"

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are GreenTrace AI, an expert sustainability auditor for Python code.

Analyze the code below and provide:
1. Sustainability issues
2. Energy optimization opportunities
3. Carbon-reduction refactoring ideas
4. Short improved-code strategy

Keep it concise, professional, and developer-focused.

Code:
{code}
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"Gemini request failed: {e}"


# -----------------------------
# UI Helpers
# -----------------------------
def status_badge(rating: str) -> str:
    css = "good" if rating == "Excellent" else "medium" if rating == "Moderate" else "bad"
    return f'<span class="badge {css}">{rating}</span>'


def metric_card(label: str, value: str, help_text: str = "") -> str:
    extra = f'<div class="small-note">{help_text}</div>' if help_text else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {extra}
    </div>
    """


def build_radar_chart(result: AnalysisResult):
    categories = ["Loops", "Nested", "Conditions", "Imports", "Repeated", "Complexity"]
    values = [
        result.num_loops,
        result.nested_loops,
        result.num_conditions,
        result.num_imports,
        result.repeated_calls,
        result.estimated_complexity_score,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name="Risk"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6fff3"),
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def build_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Green Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,99,132,0.30)"},
                    {"range": [40, 70], "color": "rgba(255,193,7,0.25)"},
                    {"range": [70, 100], "color": "rgba(62,207,142,0.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6fff3"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_hotspots_chart(hotspots: List[Tuple[str, float]]):
    if not hotspots:
        return None

    df = pd.DataFrame(hotspots, columns=["Hotspot", "Score"])
    fig = px.bar(df, x="Score", y="Hotspot", orientation="h", title="Code Hotspots")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6fff3"),
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ GreenTrace AI Settings")
    carbon_intensity = st.slider(
        "Carbon Intensity (gCO₂/kWh)",
        min_value=50,
        max_value=1000,
        value=708,
        step=10,
        help="Approximate grid carbon intensity. Higher means dirtier electricity.",
    )
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    enable_gemini = st.toggle("Enable Gemini Analysis", value=False)

    st.markdown("---")
    st.markdown("### 📁 Supported files")
    st.markdown("`.py`, `.ipynb`, `.txt`, `.md`, `.json`, `.yaml`, `.yml`, `.csv`, `.log`")


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CODE = textwrap.dedent(
    """
    import pandas as pd

    def process_data(data):
        result = []
        for i in range(len(data)):
            for j in range(len(data)):
                if data[i] > data[j]:
                    result.append(data[i] * 2)
        return result

    values = [4, 8, 3, 9, 1]
    output = process_data(values)
    print(output)
    """
).strip()


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🌿 GreenTrace AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced sustainability auditor for Python code from .py, .ipynb, markdown blocks, text files, and more.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown('<div class="section-title">🧠 Analyze Python Content</div>', unsafe_allow_html=True)
    input_mode = st.radio("Choose input method", ["Paste Code", "Upload File"], horizontal=True)

    file_name = "manual_input.py"
    file_type = "py"
    code_text = DEFAULT_CODE

    if input_mode == "Paste Code":
        code_text = st.text_area("Paste your Python code below", value=DEFAULT_CODE, height=360)
    else:
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["py", "ipynb", "txt", "md", "json", "yaml", "yml", "csv", "log"],
            help="For notebooks, GreenTrace AI extracts code cells. For markdown/text, it detects Python code blocks when present.",
        )
        if uploaded_file is not None:
            try:
                code_text, file_name, file_type = load_code_from_uploaded_file(uploaded_file)
                st.success(f"Loaded: {file_name}")
                st.text_area("Extracted Python Preview", value=code_text, height=360)
            except Exception as e:
                st.error(str(e))
                code_text = ""
        else:
            st.text_area("Preview", value=DEFAULT_CODE, height=360)
            code_text = DEFAULT_CODE

    analyze_btn = st.button("🚀 Run Sustainability Audit", use_container_width=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### ✨ What this version supports")
    st.markdown(
        """
- Python scripts (.py)
- Jupyter notebooks (.ipynb)
- Text and markdown files
- JSON/YAML/log/CSV text inputs
- Python code blocks extracted from docs
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 📝 Note")
    st.markdown(
        """
GreenTrace AI still analyzes Python code. For non-Python files, it tries to:

1. extract Python code blocks, or  
2. treat the text as Python source.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Output
# -----------------------------
if analyze_btn:
    if not code_text.strip():
        st.error("No analyzable Python code found in the input.")
    else:
        try:
            result = analyze_code(code_text, carbon_intensity, file_name, file_type)

            st.markdown('<div class="section-title">📊 Sustainability Audit Results</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("Green Score", f"{result.green_score}/100", "Higher is better"), unsafe_allow_html=True)
                st.markdown(status_badge(result.rating), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Energy Estimate", f"{result.energy_wh} Wh", "Static heuristic estimate"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("Carbon Estimate", f"{result.carbon_g_co2} g CO₂", "Based on chosen grid intensity"), unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("Complexity Score", str(result.estimated_complexity_score), "Higher means riskier"), unsafe_allow_html=True)

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.markdown(metric_card("File", result.file_name), unsafe_allow_html=True)
            with c6:
                st.markdown(metric_card("Type", result.file_type.upper()), unsafe_allow_html=True)
            with c7:
                st.markdown(metric_card("Lines", str(result.lines_of_code)), unsafe_allow_html=True)
            with c8:
                st.markdown(metric_card("Imports", str(result.num_imports)), unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Overview", "Hotspots", "Suggestions", "AI Analysis"]
            )

            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(build_gauge(result.green_score), use_container_width=True)
                with col_b:
                    st.plotly_chart(build_radar_chart(result), use_container_width=True)

                st.markdown("### Code Metrics")
                metrics_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Functions",
                            "Classes",
                            "Loops",
                            "Nested Loops",
                            "Conditions",
                            "Comprehensions",
                            "Repeated Calls",
                            "Recursive Functions",
                        ],
                        "Value": [
                            result.num_functions,
                            result.num_classes,
                            result.num_loops,
                            result.nested_loops,
                            result.num_conditions,
                            result.num_comprehensions,
                            result.repeated_calls,
                            result.recursive_functions,
                        ],
                    }
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("### ⚠️ Potential Performance / Sustainability Hotspots")
                hotspot_fig = build_hotspots_chart(result.hotspots)
                if hotspot_fig is not None:
                    st.plotly_chart(hotspot_fig, use_container_width=True)
                else:
                    st.info("No major hotspots detected by the simple heuristic.")

                if result.hotspots:
                    for hotspot, score in result.hotspots:
                        st.markdown(f"- **{hotspot}** → risk score `{score}`")

            with tab3:
                st.markdown("### ✅ Rule-Based Green Suggestions")
                for i, suggestion in enumerate(result.suggestions, start=1):
                    st.markdown(f"{i}. {suggestion}")

            with tab4:
                st.markdown("### 🤖 Gemini Sustainability Review")
                if enable_gemini:
                    with st.spinner("Generating Gemini suggestions..."):
                        ai_text = get_gemini_suggestions(code_text, gemini_api_key)
                    st.write(ai_text)
                else:
                    st.info("Enable Gemini Analysis from the sidebar to see AI-generated sustainability suggestions.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer-box">
        <b>GreenTrace AI</b> helps developers estimate code sustainability using static heuristics,
        green scoring, hotspot detection, and optional Gemini-powered suggestions.
    </div>
    """,
    unsafe_allow_html=True,
)