"""
Savings Goal Calculator using Bisection Method
Numerical root-finding approach to calculate required monthly savings
"""

class SavingsGoalCalculator:
    def __init__(self, target_amount, years, annual_rate, compounding='monthly'):
        """
        Initialize the Savings Goal Calculator
        
        Args:
            target_amount: Target savings amount (future value)
            years: Time period in years
            annual_rate: Annual interest rate (percentage)
            compounding: Compounding frequency (default: monthly)
        """
        self.target_amount = float(target_amount)
        self.years = float(years)
        self.annual_rate = float(annual_rate)
        self.compounding = compounding
        
        # Calculate derived values
        self.total_months = int(self.years * 12)
        self.monthly_rate = (self.annual_rate / 100) / 12
        
        # Bisection parameters
        self.tolerance = 0.01  # Stop when error is below this
        self.max_iterations = 100
    
    def future_value_function(self, monthly_saving):
        """
        Calculate the difference between future value and target amount
        Formula: f(S) = S × [(1 + r)^n - 1] / r - TargetAmount
        
        Args:
            monthly_saving: Monthly savings amount (S)
            
        Returns:
            Difference between calculated FV and target amount
        """
        if self.monthly_rate == 0:
            # Special case: no interest
            fv = monthly_saving * self.total_months
        else:
            # Standard formula with interest
            fv = monthly_saving * (((1 + self.monthly_rate) ** self.total_months - 1) / self.monthly_rate)
        
        return fv - self.target_amount
    
    def bisection_method(self):
        """
        Apply Bisection Method to find the required monthly saving
        
        Returns:
            Dictionary containing:
                - monthly_saving: Required monthly saving amount
                - iterations: List of iteration details
                - num_iterations: Total number of iterations
                - converged: Whether the method converged
                - final_error: Final absolute error
                - method: Method name
        """
        # Define initial interval [a, b]
        # Lower bound: 0 (no savings)
        a = 0.0
        
        # Upper bound: target amount divided by months (worst case, no interest)
        # Add some buffer to ensure we bracket the root
        b = (self.target_amount / self.total_months) * 1.5
        
        # Check if function changes sign in the interval
        fa = self.future_value_function(a)
        fb = self.future_value_function(b)
        
        if fa * fb > 0:
            # Adjust upper bound if needed
            b = self.target_amount / self.total_months * 3
            fb = self.future_value_function(b)
            
            if fa * fb > 0:
                return {
                    'error': 'Unable to find valid interval. Function may not have a root.',
                    'converged': False
                }
        
        iterations = []
        converged = False
        
        for i in range(self.max_iterations):
            # Calculate midpoint
            c = (a + b) / 2
            fc = self.future_value_function(c)
            
            # Calculate error (width of interval)
            error = abs(b - a)
            
            # Store iteration details
            iterations.append({
                'iteration': i,
                'lower_bound': a,
                'upper_bound': b,
                'midpoint': c,
                'f_midpoint': fc,
                'error': error,
                'interval_width': b - a
            })
            
            # Check convergence
            if error < self.tolerance or abs(fc) < 0.01:
                converged = True
                break
            
            # Update interval
            if fa * fc < 0:
                # Root is in [a, c]
                b = c
                fb = fc
            else:
                # Root is in [c, b]
                a = c
                fa = fc
        
        # Final result
        monthly_saving = (a + b) / 2
        final_error = abs(self.future_value_function(monthly_saving))
        
        # Calculate total amount that will be saved
        total_contributions = monthly_saving * self.total_months
        total_interest = self.target_amount - total_contributions
        
        return {
            'monthly_saving': round(monthly_saving, 2),
            'total_contributions': round(total_contributions, 2),
            'total_interest': round(total_interest, 2),
            'target_amount': round(self.target_amount, 2),
            'iterations': iterations,
            'num_iterations': len(iterations),
            'converged': converged,
            'final_error': round(final_error, 2),
            'method': 'Bisection Method',
            'monthly_rate': round(self.monthly_rate * 100, 4),
            'total_months': self.total_months,
            'annual_rate': self.annual_rate
        }
    
    def calculate(self):
        """
        Main calculation method
        
        Returns:
            Result dictionary from bisection_method()
        """
        return self.bisection_method()