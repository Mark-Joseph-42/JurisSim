# Prompt templates for JurisSim

CLAUSE_EXTRACTION_PROMPT = """You are a legal assistant. Your task is to extract individual, testable clauses from the following draft bill.
A clause is a single rule or constraint. 
Return the clauses as a bulleted list.

DRAFT BILL:
{bill_text}

CLAUSES:
"""

RED_TEAM_PROMPT = """You are an adversarial legal expert. Your task is to find potential loopholes or exploitable interpretations of the given legal clause, considering the provided context of existing laws.
Focus on how a corporation or individual might bypass the intent of the law while strictly following the letter of the law.
List 2-3 specific adversarial hypotheses.

LEGAL CLAUSE:
{clause}

EXISTING LAWS CONTEXT:
{context}

ADVERSARIAL HYPOTHESES:
"""

PATTERN_CLASSIFICATION_PROMPT = """Given the legal clause and adversarial hypothesis, classify the type of loophole and extract the relevant parameters as JSON.

LOOPHOLE TYPES:
1. threshold_splitting - An entity splits into parts to stay under a per-entity limit.
   Parameters: cap (the numeric limit as int), entity_name (what is splitting, e.g. "subsidiary")
2. definition_gap - Something is not covered by a legal definition.
   Parameters: defined_term (the term defined in law), undefined_term (the term being used instead), consequence (the legal rule/tax/penalty being avoided)
3. jurisdiction_evasion - Routing through a jurisdiction not covered by the law.
   Parameters: covered_jurisdiction, uncovered_jurisdiction, obligation (the legal requirement being avoided)
4. temporal_gap - Exploiting timing windows or deadlines.
   Parameters: event (the starting event), deadline_hours (int), loophole_action (the action taken after deadline)
5. scope_limitation - Law only applies to specific entities meeting a threshold.
   Parameters: entity_type, threshold_param (e.g. "turnover"), threshold_value (int), obligation
6. aggregation_evasion - Splitting a large value into many small ones to stay under an individual cap.
   Parameters: cap (int), item_name (e.g. "transaction"), total_value (int)
7. consent_loophole - Arguing implicit action satisfies explicit consent requirements.
   Parameters: required_consent, alternate_action (the action argued as implicit consent)

LEGAL CLAUSE:
{clause}

ADVERSARIAL HYPOTHESIS:
{hypothesis}

Output JSON ONLY in the following format:
{{"pattern": "pattern_name", "params": {{"param1": value, ...}}}}

If it doesn't fit any pattern, output {{"pattern": "none"}}.
"""

FORMALIZATION_PROMPT = """You are an expert legal formalization engine. Convert the given legal clause and adversarial hypothesis into executable Python code using the z3-solver library.
The goal is to mathematically prove if the hypothesis represents a loophole that satisfies the constraints in the text.

### EXAMPLES

Example 1: Definition Gap (Tax avoidance)
Clause: Every recognized transaction shall pay tax. Recognized transactions use national currency.
Hypothesis: A dealer uses loyalty points instead of currency.
Z3 Code:
from z3 import *
is_currency = Bool('is_currency')
is_loyalty_points = Bool('is_loyalty_points')
is_taxable = Bool('is_taxable')
tax_paid = Bool('tax_paid')
s = Solver()
s.add(Implies(is_currency, is_taxable))
s.add(Implies(is_taxable, tax_paid))
s.add(Not(Implies(is_loyalty_points, is_currency)))
s.add(is_loyalty_points == True)
s.add(is_currency == False)
s.add(tax_paid == False)
if s.check() == sat:
    print("sat")
else:
    print("unsat")

Example 2: Threshold Splitting (Carbon Cap)
Clause: A facility's carbon emissions must not exceed 1000 tons.
Hypothesis: Corporation splits into 3 subsidiaries to emit 2400 tons.
Z3 Code:
from z3 import *
num_subsidiaries = Int('num_subsidiaries')
emissions_per_sub = Int('emissions_per_sub')
cap = IntVal(1000)
total = Int('total')
s = Solver()
s.add(emissions_per_sub <= cap)
s.add(num_subsidiaries >= 2)
s.add(total == num_subsidiaries * emissions_per_sub)
s.add(total > cap)
s.add(emissions_per_sub > 0)
if s.check() == sat:
    print("sat")
else:
    print("unsat")

Example 3: Jurisdiction Evasion (Data Privacy)
Clause: Domestic data controllers must notify subjects of breaches.
Hypothesis: A company processes data through a foreign proxy.
Z3 Code:
from z3 import *
is_domestic = Bool('is_domestic')
is_foreign = Bool('is_foreign')
must_notify = Bool('must_notify')
s = Solver()
s.add(Implies(is_domestic, must_notify))
s.add(is_foreign == True)
s.add(is_domestic == False)
s.add(must_notify == False)
if s.check() == sat:
    print("sat")
else:
    print("unsat")

### TASK

LEGAL CLAUSE:
{clause}

ADVERSARIAL HYPOTHESIS:
{hypothesis}

EXISTING LAWS CONTEXT:
{context}

Output ONLY valid Python code. No markdown.

Z3 CODE:
"""

PATCH_GENERATION_PROMPT = """You are a legislative counsel. A loophole has been mathematically proven in the following clause.
Suggest a specific amendment to the clause text to close the loophole and prevent the adversarial interpretation.

LEGAL CLAUSE:
{clause}

LOOPHOLE PROOF (Z3 CODE):
{z3_code}

ADVERSARIAL HYPOTHESIS:
{hypothesis}

SUGGESTED AMENDMENT:
"""

AMBIGUITY_SCORING_PROMPT = """Rate the ambiguity of this legal clause on a scale from 0.0 to 1.0.

0.0 = Perfectly unambiguous: every term is defined, scope is explicit, no loopholes possible.
1.0 = Extremely ambiguous: key terms undefined, scope unclear, easily exploitable.

Consider: (1) Are all key terms explicitly defined? (2) Does it cover all entity types? (3) Are deadlines explicit? (4) Does it cover cross-border scenarios? (5) Are penalties specific?

CLAUSE:
{clause}

CONTEXT FROM EXISTING LAW:
{context}

Output ONLY a single decimal number between 0.0 and 1.0. Nothing else.

AMBIGUITY SCORE:
"""
