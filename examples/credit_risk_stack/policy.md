# Policies

Governance, risk and controls stacked onto the runtime and its workflows.
Each heading names a policy; the prose is the statement an LLM judges in
natural language. `Fallback:` gives a deterministic expression enforced
when no model is configured, and `Applies to:` maps the policy onto the
runtime or onto a named workflow.

## Loan Amount Cap

The requested loan amount must not exceed $75,000.

Fallback: loan_amount <= 75000
Applies to: Underwriting Workflow

## Debt-to-Income Ceiling

The applicant's debt-to-income ratio must not exceed 0.43.

Fallback: debt_to_income <= 0.43
Applies to: runtime

## Fair Lending Control

Decisions must rest only on the financial facts carried in the intent's
context -- never on age, gender, ethnicity or any other protected
attribute.

Applies to: runtime
