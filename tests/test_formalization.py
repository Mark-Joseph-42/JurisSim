import pytest
import os
from src.llm_inference import LegalLLM
from src.z3_solver import LogicSolver

@pytest.fixture(scope="module")
def llm():
    return LegalLLM()

@pytest.fixture(scope="module")
def solver():
    return LogicSolver()

def test_extract_clauses(llm):
    bill = """# Test Bill
1. Users must be 18 to join.
2. Corporations must pay 10% tax.
3. No smoking in public.
"""
    clauses = llm.extract_clauses(bill)
    assert len(clauses) >= 1
    assert any("18" in c for c in clauses)

def test_formalize_carbon_scenario(llm, solver):
    clause = "A corporation's carbon emissions per facility must not exceed 1000 tons per year."
    hypothesis = "The corporation splits into multiple subsidiaries to bypass the 1000 ton limit."
    context = "A facility is defined as a single manufacturing plant operated by a corporation."
    
    z3_code = llm.formalize_to_z3(clause, hypothesis, context)
    print("\n[Generated Z3 Code]:\n", z3_code)
    
    result = solver.verify_code(z3_code)
    # Success means it runs without error, even if it's not logically perfect yet
    assert result['result'] != 'error', f"Z3 code error: {result['stderr']}"
    assert result['result'] in ('sat', 'unsat')
