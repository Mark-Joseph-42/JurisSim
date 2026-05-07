import gradio as gr
import os
from dotenv import load_dotenv
from src.vector_db import VectorDB
from src.llm_inference import LegalLLM
from src.z3_solver import LogicSolver
from src.pipeline import analyze_bill, format_report_markdown

# Load environment variables
load_dotenv()

# Global components (initialized on first load)
db = None
llm = None
solver = None

def init_components():
    global db, llm, solver
    if db is None:
        db = VectorDB()
        db.index_all_mock_data("mock_data")
        
        use_api = os.environ.get("USE_API", "false").lower() == "true"
        if use_api:
            from src.llm_inference import LegalLLM_API
            llm = LegalLLM_API()
        else:
            llm = LegalLLM()
            
        solver = LogicSolver()

def run_analysis(bill_text):
    init_components()
    report = analyze_bill(bill_text, db, llm, solver)
    md_report = format_report_markdown(report)
    
    # Extract the first Z3 code for display
    z3_code = ""
    for clause in report['clauses']:
        if clause['loopholes']:
            z3_code = clause['loopholes'][0]['z3_code']
            break
            
    return report['score'], md_report, z3_code

# Custom CSS for premium look
custom_css = """
.container { max-width: 1200px; margin: auto; }
.header { text-align: center; margin-bottom: 2rem; }
.score-box { font-size: 2rem; font-weight: bold; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🏛️ JurisSim — Legislative Integrity Analyzer", elem_classes="header")
    gr.Markdown("Upload a draft bill to detect loopholes with mathematical certainty using neuro-symbolic AI.")
    
    with gr.Row():
        with gr.Column(scale=1):
            bill_input = gr.Textbox(
                label="Draft Bill Text", 
                lines=15, 
                placeholder="Paste your draft bill here (e.g., Clause 1: Carbon emissions must not exceed 1000 tons...)",
                value="# Draft Environmental Bill\n\n## Clause 1: Emissions Cap\nA corporation's carbon emissions per facility must not exceed 1000 tons per year."
            )
            analyze_btn = gr.Button("🔍 Analyze Bill", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Group():
                score_display = gr.Number(label="Statutory Integrity Score (RQ)", precision=2)
                report_output = gr.Markdown(label="Analysis Report")
            
            with gr.Accordion("Mathematical Proof (Z3 Code)", open=False):
                z3_output = gr.Code(label="Z3 Proof", language="python")
    
    analyze_btn.click(
        fn=run_analysis, 
        inputs=bill_input, 
        outputs=[score_display, report_output, z3_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css, server_port=7861, share=True)
