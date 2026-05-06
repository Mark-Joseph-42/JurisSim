import os
from dotenv import load_dotenv
from src.vector_db import VectorDB
from src.llm_inference import LegalLLM
from src.z3_solver import LogicSolver
from src.pipeline import analyze_bill, format_report_markdown

# Load environment variables
load_dotenv()

def main():
    print("--- JurisSim End-to-End Pipeline Prototype ---")
    
    # 1. Initialize components
    db = VectorDB()
    db.index_all_mock_data("mock_data")
    
    use_api = os.environ.get("USE_API", "false").lower() == "true"
    if use_api:
        from src.llm_inference import LegalLLM_API
        llm = LegalLLM_API()
    else:
        llm = LegalLLM()
        
    solver = LogicSolver()
    
    # 2. Define a "Draft Bill" to test
    # Try to load the demo bill if it exists, otherwise use hardcoded sample
    demo_bill_path = "demo_bills/digital_privacy_act_draft.md"
    if os.path.exists(demo_bill_path):
        with open(demo_bill_path, 'r') as f:
            draft_bill = f.read()
        print(f"Loaded demo bill from {demo_bill_path}")
    else:
        draft_bill = """
# Draft Environmental and Safety Bill 2026

## Clause 1: Corporate Carbon Responsibility
A corporation's carbon emissions per facility must not exceed 1000 tons per calendar year. Any facility exceeding this limit will be subject to a fine of $500 per additional ton.

## Clause 2: User Access Eligibility
All individuals wishing to access the national digital portal must be at least 18 years of age at the time of registration.
"""
    
    # 3. Run analysis
    report = analyze_bill(draft_bill, db, llm, solver)
    
    # 4. Output results
    print("\n" + "="*50)
    print("FINAL ANALYSIS REPORT")
    print("="*50)
    md_report = format_report_markdown(report, use_emoji=False)
    print(md_report)

if __name__ == "__main__":
    main()
