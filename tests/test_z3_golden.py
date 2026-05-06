from z3 import *
import pytest
from src.z3_solver import LogicSolver

def test_carbon_emissions_loophole():
    # Scenario: Corporation splits into subsidiaries to bypass per-facility cap
    num_subsidiaries = Int('num_subsidiaries')
    emissions_per_sub = Int('emissions_per_sub')
    cap_per_facility = IntVal(1000)
    total_emissions = Int('total_emissions')

    s = Solver()

    # Law constraint: each facility <= 1000 tons
    s.add(emissions_per_sub <= cap_per_facility)

    # Adversarial scenario: split into N subsidiaries
    s.add(num_subsidiaries >= 2)
    s.add(total_emissions == num_subsidiaries * emissions_per_sub)

    # Exploit: total emissions exceed what a single entity could emit
    s.add(total_emissions > cap_per_facility)

    # Each subsidiary is technically compliant
    s.add(emissions_per_sub > 0)

    result = s.check()
    assert result == sat
    if result == sat:
        m = s.model()
        print(f"\n[Carbon Loophole] Subsidiaries: {m[num_subsidiaries]}, Each emits: {m[emissions_per_sub]}, Total: {m[total_emissions]}")

def test_tax_digital_invoicing_loophole():
    # Scenario: Using loyalty points which are not defined as 'currency' to avoid tax
    is_currency = Bool('is_currency')
    is_loyalty_points = Bool('is_loyalty_points')
    is_taxable_transaction = Bool('is_taxable_transaction')
    tax_paid = Bool('tax_paid')

    s = Solver()

    # Law: transactions using "currency" are taxable
    s.add(Implies(is_currency, is_taxable_transaction))
    s.add(Implies(is_taxable_transaction, tax_paid))

    # Law does NOT define loyalty points as currency
    s.add(Not(Implies(is_loyalty_points, is_currency)))

    # Adversarial: use loyalty points for a transaction
    s.add(is_loyalty_points == True)
    s.add(is_currency == False)

    # Can we avoid tax?
    s.add(tax_paid == False)

    result = s.check()
    assert result == sat
    if result == sat:
        print("\n[Tax Loophole] Possible to avoid tax using loyalty points.")

def test_age_verification_no_loophole():
    # Scenario: Baseline "no loophole" case
    age = Int('age')
    registered = Bool('registered')

    s = Solver()

    # Law: must be >= 18 to register
    s.add(Implies(registered, age >= 18))

    # Try to register at age 15
    s.add(registered == True)
    s.add(age == 15)

    result = s.check()
    assert result == unsat
    print("\n[Age Verification] Law is airtight (unsat for registration at age 15).")

def test_solver_integration():
    # Test our LogicSolver class with the carbon loophole code
    solver = LogicSolver()
    code = """
from z3 import *
num_subsidiaries = Int('num_subsidiaries')
emissions_per_sub = Int('emissions_per_sub')
cap_per_facility = IntVal(1000)
total_emissions = Int('total_emissions')
s = Solver()
s.add(emissions_per_sub <= cap_per_facility)
s.add(num_subsidiaries >= 2)
s.add(total_emissions == num_subsidiaries * emissions_per_sub)
s.add(total_emissions > cap_per_facility)
s.add(emissions_per_sub > 0)
if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""
    result = solver.verify_code(code)
    assert result['result'] == 'sat'
