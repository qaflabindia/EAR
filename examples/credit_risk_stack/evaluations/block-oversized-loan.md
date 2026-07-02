# Underwrite a $200,000 personal loan far above the cap

The request is well above the $75,000 Loan Amount Cap, whatever the
applicant's quality.

## Context

- loan_amount: 200000
- credit_score: 820
- debt_to_income: 0.10

## Expected

The cycle must be blocked by governance before any underwriting happens --
the Loan Amount Cap is the violated policy.

- decision: Loan Amount Cap
