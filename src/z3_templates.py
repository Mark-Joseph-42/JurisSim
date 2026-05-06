# Z3 Template Library for JurisSim
# Each function returns a string of executable Python code.

def render_threshold_splitting(cap: int, entity_name: str = "subsidiary") -> str:
    # Fix naming: subsidiary -> subsidiaries
    var_name = f"num_{entity_name}s" if not entity_name.endswith('y') else f"num_{entity_name[:-1]}ies"
    return f"""from z3 import *
{var_name} = Int('{var_name}')
output_per_{entity_name} = Int('output_per_{entity_name}')
cap = IntVal({cap})
total = Int('total')
s = Solver()
s.add(output_per_{entity_name} <= cap)
s.add(output_per_{entity_name} > 0)
s.add({var_name} >= 2)
s.add(total == {var_name} * output_per_{entity_name})
s.add(total > cap)
if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_definition_gap(defined_term: str, undefined_term: str, consequence: str) -> str:
    return f"""from z3 import *
is_{defined_term} = Bool('is_{defined_term}')
is_{undefined_term} = Bool('is_{undefined_term}')
consequence = Bool('{consequence}')

s = Solver()
# The rule: if it's the defined term, the consequence follows
s.add(Implies(is_{defined_term}, consequence))

# The loophole: the undefined term is NOT the defined term
s.add(Not(Implies(is_{undefined_term}, is_{defined_term})))

# Adversarial scenario: we have the undefined term but NOT the defined term
s.add(is_{undefined_term} == True)
s.add(is_{defined_term} == False)

# Can we avoid the consequence?
s.add(consequence == False)

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_jurisdiction_evasion(covered_jurisdiction: str, uncovered_jurisdiction: str, obligation: str) -> str:
    return f"""from z3 import *
in_{covered_jurisdiction} = Bool('in_{covered_jurisdiction}')
in_{uncovered_jurisdiction} = Bool('in_{uncovered_jurisdiction}')
{obligation} = Bool('{obligation}')

s = Solver()
# Law applies in covered jurisdiction
s.add(Implies(in_{covered_jurisdiction}, {obligation}))

# Adversarial: route through uncovered jurisdiction
s.add(in_{uncovered_jurisdiction} == True)
s.add(in_{covered_jurisdiction} == False)

# Can we avoid the obligation?
s.add({obligation} == False)

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_temporal_gap(event: str, deadline_hours: int, loophole_action: str) -> str:
    return f"""from z3 import *
time_since_{event} = Int('time_since_{event}')
{loophole_action} = Bool('{loophole_action}')
is_obligated = Bool('is_obligated')

s = Solver()
# Law: obligation exists if within deadline
s.add(Implies(time_since_{event} <= {deadline_hours}, is_obligated))

# Adversarial: delay action until just after deadline
s.add(time_since_{event} > {deadline_hours})
s.add({loophole_action} == True)
s.add(is_obligated == False)

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_scope_limitation(entity_type: str, threshold_param: str, threshold_value: int, obligation: str) -> str:
    return f"""from z3 import *
{threshold_param} = Int('{threshold_param}')
is_{entity_type} = Bool('is_{entity_type}')
has_obligation = Bool('has_obligation')

s = Solver()
# Law applies only if entity meets type and threshold
s.add(Implies(And(is_{entity_type}, {threshold_param} > {threshold_value}), has_obligation))

# Adversarial: stay just below threshold
s.add(is_{entity_type} == True)
s.add({threshold_param} == {threshold_value})
s.add(has_obligation == False)

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_aggregation_evasion(cap: int, item_name: str, total_value: int) -> str:
    return f"""from z3 import *
num_{item_name}s = Int('num_{item_name}s')
value_per_{item_name} = Int('value_per_{item_name}')
total = Int('total')
cap = IntVal({cap})

s = Solver()
# Law: applies to individual items above cap
s.add(value_per_{item_name} <= cap)
s.add(num_{item_name}s >= 2)
s.add(total == num_{item_name}s * value_per_{item_name})
s.add(total == {total_value})

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

def render_consent_loophole(required_consent: str, alternate_action: str) -> str:
    return f"""from z3 import *
{required_consent}_obtained = Bool('{required_consent}_obtained')
{alternate_action}_performed = Bool('{alternate_action}_performed')
is_legal = Bool('is_legal')

s = Solver()
# Law: requires explicit consent
s.add(Implies(is_legal, {required_consent}_obtained))

# Adversarial: argue alternate action implies consent
s.add({required_consent}_obtained == False)
s.add({alternate_action}_performed == True)
s.add(is_legal == False) # Goal: bypass the law

if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""

TEMPLATES = {
    "threshold_splitting": render_threshold_splitting,
    "definition_gap": render_definition_gap,
    "jurisdiction_evasion": render_jurisdiction_evasion,
    "temporal_gap": render_temporal_gap,
    "scope_limitation": render_scope_limitation,
    "aggregation_evasion": render_aggregation_evasion,
    "consent_loophole": render_consent_loophole
}

def render(pattern: str, params: dict) -> str:
    if pattern in TEMPLATES:
        return TEMPLATES[pattern](**params)
    return ""
