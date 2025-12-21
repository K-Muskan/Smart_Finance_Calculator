import math

class NewtonRaphsonSolver:
    def __init__(self, tolerance=0.01, max_iterations=100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, func, func_derivative, initial_guess):
        iterations = []
        x = initial_guess
        iteration_count = 0
        
        while iteration_count < self.max_iterations:
            fx = func(x)
            fpx = func_derivative(x)
            
            if abs(fpx) < 1e-10:
                print(f"Warning: Derivative too small at iteration {iteration_count}")
                break
            
            x_new = x - (fx / fpx)
            error = abs(x_new - x)
            error_percent = (error / abs(x)) * 100 if x != 0 else 0
            
            iterations.append({
                'iteration': iteration_count,
                'guess': round(x_new, 4),
                'fx': round(fx, 4),
                'fpx': round(fpx, 6),
                'error': round(error_percent, 4)
            })
            
            if error < self.tolerance:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'converged': True,
                    'num_iterations': iteration_count + 1
                }
            
            x = x_new
            iteration_count += 1
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': False,
            'num_iterations': iteration_count
        }


class SecantSolver:
    def __init__(self, tolerance=0.01, max_iterations=100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, func, initial_guess1, initial_guess2):
        iterations = []
        x0 = initial_guess1
        x1 = initial_guess2
        iteration_count = 0
        
        while iteration_count < self.max_iterations:
            fx0 = func(x0)
            fx1 = func(x1)
            
            if abs(fx1 - fx0) < 1e-10:
                print(f"Warning: Denominator too small at iteration {iteration_count}")
                break
            
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            error = abs(x_new - x1)
            error_percent = (error / abs(x1)) * 100 if x1 != 0 else 0
            
            iterations.append({
                'iteration': iteration_count,
                'guess': round(x_new, 4),
                'fx': round(fx1, 4),
                'error': round(error_percent, 4)
            })
            
            if error < self.tolerance:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'converged': True,
                    'num_iterations': iteration_count + 1
                }
            
            x0 = x1
            x1 = x_new
            iteration_count += 1
        
        return {
            'root': x1,
            'iterations': iterations,
            'converged': False,
            'num_iterations': iteration_count
        }


class LoanCalculator:
    def __init__(self, principal, annual_rate, loan_term, 
                 term_unit='years', compounding='monthly'):
        self.principal = principal
        self.annual_rate = annual_rate
        self.loan_term = loan_term
        self.term_unit = term_unit
        self.compounding = compounding
        
        self.total_months = self._calculate_total_months()
        self.monthly_rate = self._calculate_monthly_rate()
    
    def _calculate_total_months(self):
        if self.term_unit == 'years':
            return int(self.loan_term * 12)
        return int(self.loan_term)
    
    def _calculate_monthly_rate(self):
        annual_decimal = self.annual_rate / 100
        
        if self.compounding == 'monthly':
            return annual_decimal / 12
        elif self.compounding == 'quarterly':
            return math.pow(1 + annual_decimal / 4, 1/3) - 1
        else:
            return math.pow(1 + annual_decimal, 1/12) - 1
    
    def _loan_function(self, M):
        #according to labfole --> this formula is suitable f(M) = P - M * [(1 - (1+r)^-n) / r]
        P = self.principal
        r = self.monthly_rate
        n = self.total_months
        
        if r == 0:
            return P - M * n
        
        factor = (1 - math.pow(1 + r, -n)) / r
        return P - M * factor
    
    def _loan_derivative(self, M):
        #according to labfole --> this formula is suitable f'(M) = -[(1 - (1+r)^-n) / r]
        r = self.monthly_rate
        n = self.total_months
        
        if r == 0:
            return -n
        
        return -((1 - math.pow(1 + r, -n)) / r)
    
    def calculate_newton_raphson(self, tolerance=0.01, max_iterations=100):
        initial_guess = self.principal * 0.015
        
        solver = NewtonRaphsonSolver(tolerance=tolerance, max_iterations=max_iterations)
        result = solver.solve(
            func=self._loan_function,
            func_derivative=self._loan_derivative,
            initial_guess=initial_guess
        )
        
        monthly_payment = result['root']
        total_paid = monthly_payment * self.total_months
        total_interest = total_paid - self.principal
        
        return {
            'monthlyPayment': round(monthly_payment, 2),
            'totalPaid': round(total_paid, 2),
            'totalInterest': round(total_interest, 2),
            'iterations': result['iterations'],
            'numIterations': result['num_iterations'],
            'converged': result['converged'],
            'method': 'Newton-Raphson Method',
            'monthlyRate': self.monthly_rate,
            'totalMonths': self.total_months,
            'principalAmount': self.principal,
            'annualRate': self.annual_rate
        }
    
    def calculate_secant(self, tolerance=0.01, max_iterations=100):
        initial_guess1 = self.principal * 0.015
        initial_guess2 = self.principal * 0.02
        
        solver = SecantSolver(tolerance=tolerance, max_iterations=max_iterations)
        result = solver.solve(
            func=self._loan_function,
            initial_guess1=initial_guess1,
            initial_guess2=initial_guess2
        )
        
        monthly_payment = result['root']
        total_paid = monthly_payment * self.total_months
        total_interest = total_paid - self.principal
        
        return {
            'monthlyPayment': round(monthly_payment, 2),
            'totalPaid': round(total_paid, 2),
            'totalInterest': round(total_interest, 2),
            'iterations': result['iterations'],
            'numIterations': result['num_iterations'],
            'converged': result['converged'],
            'method': 'Secant Method',
            'monthlyRate': self.monthly_rate,
            'totalMonths': self.total_months,
            'principalAmount': self.principal,
            'annualRate': self.annual_rate
        }
    
    def generate_amortization_schedule(self, monthly_payment, monthly_rate, total_months):
        schedule = []
        balance = self.principal
        
        for month in range(1, total_months + 1):
            interest_payment = balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            balance -= principal_payment
            
            if month == total_months:
                balance = 0
            
            schedule.append({
                'month': month,
                'payment': round(monthly_payment, 2),
                'principal': round(principal_payment, 2),
                'interest': round(interest_payment, 2),
                'balance': round(max(0, balance), 2)
            })
        
        return schedule
    
    def get_loan_summary(self):
        return {
            'principal': self.principal,
            'annualRate': self.annual_rate,
            'loanTerm': self.loan_term,
            'termUnit': self.term_unit,
            'compounding': self.compounding,
            'totalMonths': self.total_months,
            'monthlyRate': round(self.monthly_rate * 100, 4),
            'monthlyRatePercent': f"{self.monthly_rate * 100:.4f}%"
        }


if __name__ == '__main__':
    pass