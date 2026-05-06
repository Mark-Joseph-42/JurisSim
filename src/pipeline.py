from src.vector_db import VectorDB
from src.llm_inference import LegalLLM
from src.z3_solver import LogicSolver

def analyze_bill(bill_text: str, db: VectorDB, llm: LegalLLM, solver: LogicSolver) -> dict:
    report = {"clauses": [], "score": 0.0}
    
    # 1. Extract clauses
    print("Extracting clauses from bill...")
    clauses = llm.extract_clauses(bill_text)
    print(f"Extracted {len(clauses)} clauses.")
    
    for i, clause in enumerate(clauses):
        print(f"\nAnalyzing Clause {i+1}/{len(clauses)}: {clause[:100]}...")
        clause_report = {"text": clause, "loopholes": [], "status": "secure", "rq_score": 0.0}
        
        # 2. Retrieve relevant existing laws
        print("Retrieving context from Vector DB...")
        context_docs = db.search(clause, top_k=3)
        context = "\n---\n".join([d['text'] for d in context_docs])

        # 2.5 Score Ambiguity
        print("Scoring clause ambiguity...")
        rq_score = llm.score_ambiguity(clause, context)
        clause_report["rq_score"] = rq_score
        print(f"Ambiguity Score (RQ): {rq_score:.2f}")

        # Skip if clause is very secure (optional but recommended for speed)
        # if rq_score < 0.2:
        #     print("Clause appears highly secure. Skipping deep red-teaming.")
        #     report["clauses"].append(clause_report)
        #     continue
        
        # 3. Red-team: generate adversarial hypotheses
        print("Generating adversarial hypotheses...")
        hypotheses = llm.red_team_clause(clause, context)
        print(f"Generated {len(hypotheses)} hypotheses.")
        
        for h_idx, hypothesis in enumerate(hypotheses):
            print(f"  Verifying Hypothesis {h_idx+1}: {hypothesis[:80]}...")
            
            # 4. Formalize to Z3
            z3_code = llm.formalize_to_z3(clause, hypothesis, context)
            
            # Skip if code is clearly garbage
            if len(z3_code) < 30:
                print(f"  [?] Inconclusive: LLM generated invalid/too short Z3 code.")
                if clause_report["status"] == "secure":
                    clause_report["status"] = "inconclusive"
                continue

            # 5. Verify
            result = solver.verify_code(z3_code)
            
            if result['result'] == 'sat':
                print(f"  [!] LOOPHOLE DETECTED: {hypothesis}")
                # 6. Loophole confirmed — generate patch
                patch = llm.generate_patch(clause, hypothesis, z3_code)
                clause_report["loopholes"].append({
                    "hypothesis": hypothesis,
                    "z3_code": z3_code,
                    "proof": result['stdout'],
                    "patch": patch
                })
                clause_report["status"] = "vulnerable"
            elif result['result'] == 'error':
                print(f"  [?] Solver Error: {result['stderr'][:100]}")
                if clause_report["status"] == "secure":
                    clause_report["status"] = "inconclusive"
            else:
                print(f"  [+] Scenario appears secure.")
        
        report["clauses"].append(clause_report)
    
    # 7. Compute integrity score
    total = len(report["clauses"])
    if total > 0:
        secure = sum(1 for c in report["clauses"] if c["status"] == "secure")
        report["score"] = secure / total
    else:
        report["score"] = 1.0
    
    return report

def format_report_markdown(report: dict, use_emoji: bool = True) -> str:
    md = f"# Legislative Integrity Report\n\n"
    md += f"**Overall Integrity Score (RQ): {report['score']:.2f}**\n\n"
    
    icons = {
        "secure": "✅" if use_emoji else "[OK]",
        "vulnerable": "🚨" if use_emoji else "[!!]",
        "inconclusive": "❓" if use_emoji else "[??]"
    }
    
    for i, c in enumerate(report['clauses']):
        status_icon = icons.get(c['status'], icons["inconclusive"])
        md += f"## {status_icon} Clause {i+1} ({c['status'].capitalize()})\n"
        md += f"**Ambiguity Score (RQ): {c.get('rq_score', 0.0):.2f}**\n"
        md += f"> {c['text']}\n\n"
        
        if c['loopholes']:
            md += "### Detected Loopholes\n"
            for l in c['loopholes']:
                md += f"- **Hypothesis**: {l['hypothesis']}\n"
                md += f"- **Proof (Z3 Result)**: `sat`\n"
                md += f"- **Suggested Patch**: {l['patch']}\n\n"
        elif c['status'] == 'inconclusive':
            md += "Analysis was inconclusive due to formalization errors. Further review required.\n\n"
        else:
            md += "No loopholes detected for this clause under tested hypotheses.\n\n"
    
    return md
