import json
import os
import random

def generate_expanded_data():
    base_data = []
    # Seed data from the original 13 rows if they exist
    if os.path.exists("training/dataset.jsonl"):
        with open("training/dataset.jsonl", 'r') as f:
            for line in f:
                if line.strip():
                    base_data.append(json.loads(line))
    
    # DPDP Substantive Examples
    dpdp_clauses = [
        {
            "clause": "Section 2(i): 'Data Fiduciary' means any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data.",
            "extraction": "- Mandatory Requirement: Data Fiduciary identification based on determination of purpose and means of processing.",
            "loopholes": "Adversarial Analysis for Definition Gap: An entity might claim to be a 'Data Processor' (not a Fiduciary) by arguing that a parent company or a DAO (Decentralized Autonomous Organization) technically 'determines' the purpose, while the entity merely executes. This evades the primary liabilities and notice requirements of a Fiduciary.",
            "z3": "from z3 import *\n\nis_fiduciary = Bool('is_fiduciary')\ndetermines_purpose = Bool('determines_purpose')\ndetermines_means = Bool('determines_means')\nentity_is_processor = Bool('entity_is_processor')\n\ns = Solver()\n# Law: determines purpose AND means -> Fiduciary\ns.add(Implies(And(determines_purpose, determines_means), is_fiduciary))\n\n# Adversarial: processor determines means but NOT purpose (claims purpose comes from code/DAO)\ns.add(determines_means == True)\ns.add(determines_purpose == False)\ns.add(is_fiduciary == False)\n\nif s.check() == sat:\n    print('sat')"
        },
        {
            "clause": "Section 4: A person may process the personal data of a Data Principal only in accordance with the provisions of this Act and for a lawful purpose for which the Data Principal has given her consent.",
            "extraction": "- Mandatory Requirement: Lawful purpose and explicit consent for data processing.",
            "loopholes": "Adversarial Analysis for Consent Loophole: A service provider could use 'Dark Patterns' to obtain consent through implicit actions, like 'by scrolling you agree' or 'pre-ticked boxes', arguing that the physical act of usage constitutes consent for all lawful purposes under Section 4.",
            "z3": "from z3 import *\n\nconsent_given = Bool('consent_given')\nuser_scrolled = Bool('user_scrolled')\nexplicit_action = Bool('explicit_action')\n\ns = Solver()\n# Law requires consent\ns.add(consent_given == True)\n# Adversarial claim: scrolling implies consent\ns.add(Implies(user_scrolled, consent_given))\ns.add(user_scrolled == True)\ns.add(explicit_action == False)\n\nif s.check() == sat:\n    print('sat')"
        }
    ]

    for d in dpdp_clauses:
        base_data.append({"instruction": "Extract testable clauses from this legal text.", "input": d['clause'], "response": d['extraction']})
        base_data.append({"instruction": "Find adversarial interpretations of this clause.", "input": d['clause'], "response": d['loopholes']})
        base_data.append({"instruction": "Formalize this legal clause and adversarial hypothesis as Z3 code.", "input": f"Clause: {d['clause']}\nHypothesis: {d['loopholes'].split(':')[0]}", "response": d['z3']})

    # Pattern Generation for diverse Acts
    acts = {
        "IT Act 2000": "Section 43A regarding compensation for failure to protect sensitive personal data.",
        "BNS 2023": "Provisions regarding criminal breach of trust and misappropriation of property.",
        "Companies Act 2013": "Section 135 mandates CSR spending (2% of net profit) for companies above thresholds.",
        "Income Tax Act 1961": "Section 115JB regarding Minimum Alternate Tax (MAT) on book profits.",
        "FDI Policy": "Restrictions on multi-brand retail and inventory-based e-commerce models."
    }
    
    patterns = {
        "Threshold Splitting": {
            "analysis": "Adversarial Analysis for Threshold Splitting: A corporation can split into multiple subsidiaries, each staying just below the numeric threshold (e.g., turnover or profit) to avoid mandatory obligations like CSR spending or audit requirements.",
            "amendment": "Amended Text: 'For the purpose of calculating thresholds, the aggregate turnover/profit of all related parties, subsidiaries, and group entities as defined under Section 2(76) shall be considered as a single unit.'"
        },
        "Definition Gap": {
            "analysis": "Adversarial Analysis for Definition Gap: Exploiting the lack of specific mention of 'Virtual Digital Assets' or 'Stablecoins' in older definitions of 'recognized currency' or 'securities' to avoid taxation or regulatory oversight.",
            "amendment": "Amended Text: 'The definition of \"currency\" and \"securities\" shall include all digital assets, tokens, and cryptographic representations of value, regardless of the underlying technology or platform.'"
        },
        "Jurisdiction Evasion": {
            "analysis": "Adversarial Analysis for Jurisdiction Evasion: Routing transactions or data processing through a 'Mirror Proxy' in a tax haven or a country with lax privacy laws, arguing that the primary 'nexus' of the operation is outside the local jurisdiction.",
            "amendment": "Amended Text: 'The provisions of this Act apply to any entity that provides services to residents within the territory, regardless of the physical location of the server or the registration of the entity.'"
        }
    }

    # Generate substantive variations
    for act_name, act_text in acts.items():
        for pat_name, pat_content in patterns.items():
            base_data.append({
                "instruction": f"Analyze this clause from {act_name} for a {pat_name} loophole.",
                "input": act_text,
                "response": pat_content["analysis"]
            })
            base_data.append({
                "instruction": "Suggest an amendment to close this loophole.",
                "input": f"Loophole in {act_name} involving {pat_name}.",
                "response": pat_content["amendment"]
            })

    # Save expanded data
    with open("training/dataset.jsonl", 'w') as f:
        for item in base_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Dataset expanded to {len(base_data)} rows.")

if __name__ == "__main__":
    generate_expanded_data()
