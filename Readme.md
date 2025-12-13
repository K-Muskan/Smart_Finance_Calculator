1. Newton-Raphson Method (Based on Lab Manual):

✅ Formula: x(n+1) = x(n) - f(x(n)) / f'(x(n))
✅ Single initial guess
✅ Function and derivative evaluation
✅ Iterative refinement until convergence
✅ Tolerance-based stopping criteria

2. Loan Calculation:

✅ Solves: f(M) = P - M * [(1 - (1+r)^-n) / r] = 0
✅ Derivative: f'(M) = -[(1 - (1+r)^-n) / r]
✅ Handles different compounding frequencies
✅ Generates amortization schedules

3. Flask Integration:

✅ /api/calculate - Calculate payment
✅ /api/amortization - Get schedule
✅ /api/export-csv - Export data
✅ Error handling and validation