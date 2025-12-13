"""
Loan Payment Calculator using Newton-Raphson Method
Based on Numerical Analysis Lab Manual - Bahria University

This module implements the Newton-Raphson method for calculating
loan payments by solving the loan payment equation numerically.

Newton-Raphson Formula:
x(n+1) = x(n) - f(x(n)) / f'(x(n))

Where:
- x(n) is the current approximation
- f(x(n)) is the function value at x(n)
- f'(x(n)) is the derivative of the function at x(n)

For loan calculations:
- x represents the monthly payment M
- f(M) = P - M * [(1 - (1+r)^-n) / r]
- f'(M) = -[(1 - (1+r)^-n) / r]

Author: Khadija Muskan
Course: Numerical Analysis
Module: 2 - Loan Payment Analyzer
"""

import math


class NewtonRaphsonSolver:
    """
    Newton-Raphson method implementation for root finding
    
    The Newton-Raphson method is an open method that uses:
    1. A single initial guess
    2. Function evaluation f(x)
    3. Derivative evaluation f'(x)
    
    It typically converges faster than bracketing methods.
    """
    
    def __init__(self, tolerance=0.01, max_iterations=100):
        """
        Initialize Newton-Raphson solver
        
        Args:
            tolerance (float): Convergence tolerance (default: 0.01)
            max_iterations (int): Maximum number of iterations (default: 100)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, func, func_derivative, initial_guess):
        """
        Solve for root using Newton-Raphson method
        
        Algorithm:
        1. Start with initial guess x0
        2. Calculate f(x) and f'(x)
        3. Calculate next approximation: x_new = x - f(x)/f'(x)
        4. Check convergence: |x_new - x| < tolerance
        5. Repeat until convergence or max iterations
        
        Args:
            func: Function f(x) to find root of
            func_derivative: Derivative f'(x) of the function
            initial_guess: Initial guess x0
        
        Returns:
            dict: {
                'root': Final approximation,
                'iterations': List of iteration details,
                'converged': Boolean indicating convergence,
                'num_iterations': Number of iterations performed
            }
        """
        iterations = []
        x = initial_guess
        prev_x = float('inf')
        iteration_count = 0
        
        while iteration_count < self.max_iterations:
            # Calculate function value and derivative at current x
            fx = func(x)
            fpx = func_derivative(x)
            
            # Check if derivative is too small (avoid division by zero)
            if abs(fpx) < 1e-10:
                print(f"Warning: Derivative too small at iteration {iteration_count}")
                break
            
            # Calculate next approximation using Newton-Raphson formula
            # x(n+1) = x(n) - f(x(n)) / f'(x(n))
            x_new = x - (fx / fpx)
            
            # Calculate error
            error = abs(x_new - x)
            error_percent = (error / abs(x)) * 100 if x != 0 else 0
            
            # Store iteration details
            iterations.append({
                'iteration': iteration_count,
                'guess': round(x_new, 4),
                'fx': round(fx, 4),
                'fpx': round(fpx, 6),
                'error': round(error_percent, 4)
            })
            
            # Check convergence
            if error < self.tolerance:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'converged': True,
                    'num_iterations': iteration_count + 1
                }
            
            # Update for next iteration
            x = x_new
            iteration_count += 1
        
        # Max iterations reached without convergence
        return {
            'root': x,
            'iterations': iterations,
            'converged': False,
            'num_iterations': iteration_count
        }


class SecantSolver:
    """
    Secant method implementation for root finding
    
    The Secant method is similar to Newton-Raphson but:
    1. Uses two initial guesses
    2. Approximates derivative using finite difference
    3. Does not require explicit derivative
    
    Formula: x(n+1) = x(n) - f(x(n)) * (x(n) - x(n-1)) / (f(x(n)) - f(x(n-1)))
    """
    
    def __init__(self, tolerance=0.01, max_iterations=100):
        """
        Initialize Secant solver
        
        Args:
            tolerance (float): Convergence tolerance (default: 0.01)
            max_iterations (int): Maximum number of iterations (default: 100)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, func, initial_guess1, initial_guess2):
        """
        Solve for root using Secant method
        
        Algorithm:
        1. Start with two initial guesses x0 and x1
        2. Calculate f(x0) and f(x1)
        3. Calculate next approximation using secant formula
        4. Check convergence
        5. Update guesses and repeat
        
        Args:
            func: Function f(x) to find root of
            initial_guess1: First initial guess x0
            initial_guess2: Second initial guess x1
        
        Returns:
            dict: {
                'root': Final approximation,
                'iterations': List of iteration details,
                'converged': Boolean indicating convergence,
                'num_iterations': Number of iterations performed
            }
        """
        iterations = []
        x0 = initial_guess1
        x1 = initial_guess2
        iteration_count = 0
        
        while iteration_count < self.max_iterations:
            # Calculate function values
            fx0 = func(x0)
            fx1 = func(x1)
            
            # Check if denominator is too small (avoid division by zero)
            if abs(fx1 - fx0) < 1e-10:
                print(f"Warning: Denominator too small at iteration {iteration_count}")
                break
            
            # Calculate next approximation using Secant formula
            # x(n+1) = x(n) - f(x(n)) * (x(n) - x(n-1)) / (f(x(n)) - f(x(n-1)))
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            
            # Calculate error
            error = abs(x_new - x1)
            error_percent = (error / abs(x1)) * 100 if x1 != 0 else 0
            
            # Store iteration details
            iterations.append({
                'iteration': iteration_count,
                'guess': round(x_new, 4),
                'fx': round(fx1, 4),
                'error': round(error_percent, 4)
            })
            
            # Check convergence
            if error < self.tolerance:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'converged': True,
                    'num_iterations': iteration_count + 1
                }
            
            # Update for next iteration
            x0 = x1
            x1 = x_new
            iteration_count += 1
        
        # Max iterations reached without convergence
        return {
            'root': x1,
            'iterations': iterations,
            'converged': False,
            'num_iterations': iteration_count
        }


class LoanCalculator:
    """
    Loan Payment Calculator using Newton-Raphson Method
    
    This class calculates monthly loan payments by solving the
    loan payment equation using the Newton-Raphson numerical method.
    
    Loan Payment Equation:
    P = M * [(1 - (1+r)^-n) / r]
    
    Where:
    - P = Principal (loan amount)
    - M = Monthly payment (what we're solving for)
    - r = Monthly interest rate
    - n = Total number of payments
    
    We rearrange to find root:
    f(M) = P - M * [(1 - (1+r)^-n) / r] = 0
    """
    
    def __init__(self, principal, annual_rate, loan_term, 
                 term_unit='years', compounding='monthly'):
        """
        Initialize loan calculator
        
        Args:
            principal (float): Loan amount in dollars
            annual_rate (float): Annual interest rate as percentage (e.g., 5.5 for 5.5%)
            loan_term (float): Duration of loan
            term_unit (str): 'years' or 'months'
            compounding (str): 'monthly', 'quarterly', or 'annually'
        """
        self.principal = principal
        self.annual_rate = annual_rate
        self.loan_term = loan_term
        self.term_unit = term_unit
        self.compounding = compounding
        
        # Calculate derived values
        self.total_months = self._calculate_total_months()
        self.monthly_rate = self._calculate_monthly_rate()
    
    def _calculate_total_months(self):
        """
        Calculate total number of monthly payments
        
        Returns:
            int: Total number of months
        """
        if self.term_unit == 'years':
            return int(self.loan_term * 12)
        return int(self.loan_term)
    
    def _calculate_monthly_rate(self):
        """
        Calculate effective monthly interest rate based on compounding frequency
        
        Formulas:
        - Monthly: r_monthly = annual_rate / 12
        - Quarterly: r_monthly = (1 + annual_rate/4)^(1/3) - 1
        - Annually: r_monthly = (1 + annual_rate)^(1/12) - 1
        
        Returns:
            float: Monthly interest rate as decimal
        """
        annual_decimal = self.annual_rate / 100
        
        if self.compounding == 'monthly':
            return annual_decimal / 12
        elif self.compounding == 'quarterly':
            # Convert quarterly to monthly
            return math.pow(1 + annual_decimal / 4, 1/3) - 1
        else:  # annually
            # Convert annual to monthly
            return math.pow(1 + annual_decimal, 1/12) - 1
    
    def _loan_function(self, M):
        """
        Loan payment function f(M) to find root of
        
        Formula: f(M) = P - M * [(1 - (1+r)^-n) / r]
        
        Special case: If r = 0, f(M) = P - M * n
        
        Args:
            M (float): Monthly payment guess
        
        Returns:
            float: Function value at M
        """
        P = self.principal
        r = self.monthly_rate
        n = self.total_months
        
        # Handle zero interest rate case
        if r == 0:
            return P - M * n
        
        # Standard case with interest
        factor = (1 - math.pow(1 + r, -n)) / r
        return P - M * factor
    
    def _loan_derivative(self, M):
        """
        Derivative of loan payment function f'(M)
        
        Formula: f'(M) = -[(1 - (1+r)^-n) / r]
        
        Special case: If r = 0, f'(M) = -n
        
        Args:
            M (float): Monthly payment value (not used, but kept for consistency)
        
        Returns:
            float: Derivative value
        """
        r = self.monthly_rate
        n = self.total_months
        
        # Handle zero interest rate case
        if r == 0:
            return -n
        
        # Standard case with interest
        return -((1 - math.pow(1 + r, -n)) / r)
    
    def calculate_newton_raphson(self, tolerance=0.01, max_iterations=100):
        """
        Calculate monthly payment using Newton-Raphson method
        
        Process:
        1. Define loan equation and its derivative
        2. Make initial guess (1.5% of principal)
        3. Apply Newton-Raphson iteration
        4. Calculate total payment and interest
        
        Args:
            tolerance (float): Convergence tolerance
            max_iterations (int): Maximum iterations allowed
        
        Returns:
            dict: {
                'monthlyPayment': Monthly payment amount,
                'totalPaid': Total amount paid over loan term,
                'totalInterest': Total interest paid,
                'iterations': List of iteration details,
                'numIterations': Number of iterations,
                'converged': Boolean convergence status,
                'method': Method name,
                'monthlyRate': Monthly interest rate,
                'totalMonths': Total number of payments
            }
        """
        # Initial guess: approximately 1.5% of principal
        # This is typically a good starting point for most loans
        initial_guess = self.principal * 0.015
        
        # Create Newton-Raphson solver
        solver = NewtonRaphsonSolver(
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        
        # Solve for monthly payment
        result = solver.solve(
            func=self._loan_function,
            func_derivative=self._loan_derivative,
            initial_guess=initial_guess
        )
        
        # Calculate final values
        monthly_payment = result['root']
        total_paid = monthly_payment * self.total_months
        total_interest = total_paid - self.principal
        
        # Return comprehensive result
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
        """
        Calculate monthly payment using Secant method
        
        Process:
        1. Define loan equation
        2. Make two initial guesses
        3. Apply Secant iteration
        4. Calculate total payment and interest
        
        Args:
            tolerance (float): Convergence tolerance
            max_iterations (int): Maximum iterations allowed
        
        Returns:
            dict: Same structure as calculate_newton_raphson
        """
        # Two initial guesses
        initial_guess1 = self.principal * 0.015  # 1.5% of principal
        initial_guess2 = self.principal * 0.02   # 2% of principal
        
        # Create Secant solver
        solver = SecantSolver(
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        
        # Solve for monthly payment
        result = solver.solve(
            func=self._loan_function,
            initial_guess1=initial_guess1,
            initial_guess2=initial_guess2
        )
        
        # Calculate final values
        monthly_payment = result['root']
        total_paid = monthly_payment * self.total_months
        total_interest = total_paid - self.principal
        
        # Return comprehensive result
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
        """
        Generate complete amortization schedule
        
        For each month, calculate:
        - Interest payment = Current balance × Monthly rate
        - Principal payment = Monthly payment - Interest payment
        - New balance = Previous balance - Principal payment
        
        Args:
            monthly_payment (float): Monthly payment amount
            monthly_rate (float): Monthly interest rate
            total_months (int): Total number of months
        
        Returns:
            list: List of dictionaries containing monthly payment details
        """
        schedule = []
        balance = self.principal
        
        for month in range(1, total_months + 1):
            # Calculate interest payment for this month
            interest_payment = balance * monthly_rate
            
            # Calculate principal payment for this month
            principal_payment = monthly_payment - interest_payment
            
            # Update balance
            balance -= principal_payment
            
            # Handle final payment rounding (ensure balance reaches 0)
            if month == total_months:
                balance = 0
            
            # Add month details to schedule
            schedule.append({
                'month': month,
                'payment': round(monthly_payment, 2),
                'principal': round(principal_payment, 2),
                'interest': round(interest_payment, 2),
                'balance': round(max(0, balance), 2)
            })
        
        return schedule
    
    def get_loan_summary(self):
        """
        Get a summary of loan parameters
        
        Returns:
            dict: Summary of loan details
        """
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


# ============================================
# TESTING AND DEMONSTRATION
# ============================================

if __name__ == '__main__':
    """
    Test the loan calculator with example values
    """
    print("=" * 60)
    print("LOAN PAYMENT CALCULATOR - NEWTON-RAPHSON METHOD")
    print("=" * 60)
    print()
    
    # Example 1: Standard 30-year mortgage
    print("Example 1: 30-year mortgage")
    print("-" * 60)
    
    calculator1 = LoanCalculator(
        principal=50000,
        annual_rate=5.5,
        loan_term=30,
        term_unit='years',
        compounding='monthly'
    )
    
    result1 = calculator1.calculate_newton_raphson()
    
    print(f"Principal Amount: ${result1['principalAmount']:,.2f}")
    print(f"Annual Interest Rate: {result1['annualRate']}%")
    print(f"Loan Term: 30 years ({result1['totalMonths']} months)")
    print(f"\nMonthly Payment: ${result1['monthlyPayment']:,.2f}")
    print(f"Total Amount Paid: ${result1['totalPaid']:,.2f}")
    print(f"Total Interest Paid: ${result1['totalInterest']:,.2f}")
    print(f"Convergence: {result1['converged']}")
    print(f"Iterations: {result1['numIterations']}")
    print()
    
    # Example 2: Short-term car loan
    print("Example 2: 5-year car loan")
    print("-" * 60)
    
    calculator2 = LoanCalculator(
        principal=25000,
        annual_rate=4.5,
        loan_term=5,
        term_unit='years',
        compounding='monthly'
    )
    
    result2 = calculator2.calculate_newton_raphson()
    
    print(f"Principal Amount: ${result2['principalAmount']:,.2f}")
    print(f"Annual Interest Rate: {result2['annualRate']}%")
    print(f"Loan Term: 5 years ({result2['totalMonths']} months)")
    print(f"\nMonthly Payment: ${result2['monthlyPayment']:,.2f}")
    print(f"Total Amount Paid: ${result2['totalPaid']:,.2f}")
    print(f"Total Interest Paid: ${result2['totalInterest']:,.2f}")
    print(f"Convergence: {result2['converged']}")
    print(f"Iterations: {result2['numIterations']}")
    print()
    
    print("=" * 60)