import numpy as np
from typing import List, Tuple, Dict, Optional
import math

class InvestmentGrowthCalculator:
    """
    Investment Growth Estimator using Numerical Interpolation Methods
    Supports Newton's Forward/Backward, Divided Difference, and Lagrange Interpolation
    WITH DETAILED STEP-BY-STEP CALCULATIONS
    """
    
    def __init__(self, initial_investment: float, years: List[int], 
                 rates: List[float], target_year: int, 
                 recurring_contribution: float = 0,
                 contribution_frequency: str = 'yearly',
                 variable_contributions: Optional[Dict[int, float]] = None,
                 risk_factor: float = 0):
        """
        Initialize the investment calculator
        """
        self.initial_investment = initial_investment
        self.years = sorted(years)
        self.rates = rates
        self.start_year = min(years)
        self.target_year = target_year
        self.recurring_contribution = recurring_contribution
        self.contribution_frequency = contribution_frequency
        self.variable_contributions = variable_contributions or {}
        self.risk_factor = risk_factor
        
        # Detailed calculation tracking
        self.interpolation_details = []
        
        # Determine interpolation method
        self.has_equal_intervals = self._check_equal_intervals()
        self.interpolation_method = self._select_interpolation_method()
        self.h = self.years[1] - self.years[0] if len(self.years) > 1 else 1
        
    def _check_equal_intervals(self) -> bool:
        """Check if years have equal intervals"""
        if len(self.years) < 2:
            return True
        
        intervals = [self.years[i+1] - self.years[i] for i in range(len(self.years)-1)]
        return len(set(intervals)) == 1
    
    def _select_interpolation_method(self) -> str:
        """Auto-select appropriate interpolation method"""
        if self.has_equal_intervals:
            # Use the middle of the historical data range, not start to target
            data_start = min(self.years)
            data_end = max(self.years)
            mid_point = (data_start + data_end) / 2
            
            # For years closer to the start of data, use forward
            # For years closer to the end of data, use backward
            if self.target_year <= mid_point:
                return "Newton Forward"
            else:
                return "Newton Backward"
        else:
            return "Lagrange"
    
    def _calculate_u_forward(self, u: float, n: int) -> float:
        """Calculate u(u-1)(u-2)...(u-n+1) for forward interpolation"""
        result = u
        for i in range(1, n):
            result *= (u - i)
        return result
    
    def _calculate_u_backward(self, u: float, n: int) -> float:
        """Calculate u(u+1)(u+2)...(u+n-1) for backward interpolation"""
        result = u
        for i in range(1, n):
            result *= (u + i)
        return result
    
    def _factorial(self, n: int) -> int:
        """Calculate factorial of n"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def _format_difference_table(self, table: List[List[float]], method: str) -> str:
        """Format difference table as string for display"""
        result = f"\n{'='*60}\n"
        result += f"{method} DIFFERENCE TABLE\n"
        result += f"{'='*60}\n"
        
        n = len(table)
        # Header
        if method == "Forward":
            headers = ['x', 'f(x)'] + [f'Δ^{i}f(x)' for i in range(1, n)]
        elif method == "Backward":
            headers = ['x', 'f(x)'] + [f'∇^{i}f(x)' for i in range(1, n)]
        else:
            headers = ['x', 'f(x)'] + [f'f[x₀,...,x_{i}]' for i in range(1, n)]
        
        result += f"{'Year':<10}"
        for i, header in enumerate(headers[1:]):
            if i < n:
                result += f"{header:<15}"
        result += "\n" + "-"*60 + "\n"
        
        # Data rows - FIXED FOR BACKWARD METHOD
        if method == "Backward":
            # For backward differences, show from top to bottom
            # Only show non-zero differences
            for i in range(n):
                result += f"{self.years[i]:<10}"
                result += f"{table[i][0]:<15.6f}"  # Always show f(x)
                # Show differences only if they exist (row i can have up to i differences)
                for j in range(1, min(i + 1, n)):
                    if j < len(table[i]) and table[i][j] != 0:
                        result += f"{table[i][j]:<15.6f}"
                result += "\n"
        else:
            # Original logic for Forward and other methods
            for i in range(n):
                result += f"{self.years[i]:<10}"
                for j in range(n - i):
                    if j < len(table[i]):
                        result += f"{table[i][j]:<15.6f}"
                result += "\n"
        
        result += f"{'='*60}\n"
        return result
    
    def _newton_forward_interpolation(self, x_target: int) -> Tuple[float, Dict]:
        """
        Newton's Forward Interpolation with detailed steps
        """
        n = len(self.years)
        h = self.h
        
        details = {
            'method': 'Newton Forward Interpolation',
            'formula': 'f(a + uh) = f(a) + u·Δf(a) + [u(u-1)/2!]·Δ²f(a) + [u(u-1)(u-2)/3!]·Δ³f(a) + ...',
            'steps': []
        }
        
        # Step 1: Create forward difference table
        diff_table = [[0] * n for _ in range(n)]
        for i in range(n):
            diff_table[i][0] = self.rates[i]
        
        details['steps'].append(f"Step 1: Given data points")
        for i in range(n):
            details['steps'].append(f"  x₍{i}₎ = {self.years[i]}, f(x₍{i}₎) = {self.rates[i]:.4f}%")
        
        # Calculate forward differences
        details['steps'].append("")
        details['steps'].append(f"Step 2: Calculate forward differences")
        details['steps'].append(f"  Formula: Δf(xᵢ) = f(xᵢ₊₁) - f(xᵢ)")
        details['steps'].append("")
        details['steps'].append("  First order differences (Δ¹):")
        
        for j in range(1, n):
            for i in range(n - j):
                diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
                if j == 1:
                    details['steps'].append(f"    Δf(x₍{i}₎) = f(x₍{i+1}₎) - f(x₍{i}₎) = {diff_table[i+1][j-1]:.6f} - {diff_table[i][j-1]:.6f} = {diff_table[i][j]:.6f}")
        
        # Higher order differences
        if n > 2:
            details['steps'].append("")
            details['steps'].append("  Second order differences (Δ²):")
            for i in range(n - 2):
                details['steps'].append(f"    Δ²f(x₍{i}₎) = Δf(x₍{i+1}₎) - Δf(x₍{i}₎) = {diff_table[i+1][1]:.6f} - {diff_table[i][1]:.6f} = {diff_table[i][2]:.6f}")
        
        if n > 3:
            details['steps'].append("")
            details['steps'].append("  Third order differences (Δ³):")
            for i in range(n - 3):
                details['steps'].append(f"    Δ³f(x₍{i}₎) = Δ²f(x₍{i+1}₎) - Δ²f(x₍{i}₎) = {diff_table[i+1][2]:.6f} - {diff_table[i][2]:.6f} = {diff_table[i][3]:.6f}")
        
        # Add difference table
        details['difference_table'] = self._format_difference_table(diff_table, "Forward")
        
        # Step 3: Calculate u
        details['steps'].append("")
        details['steps'].append(f"Step 3: Calculate u")
        details['steps'].append(f"  Formula: u = (x - x₀) / h")
        details['steps'].append(f"  Where:")
        details['steps'].append(f"    x = {x_target} (target year)")
        details['steps'].append(f"    x₀ = {self.years[0]} (first data point)")
        details['steps'].append(f"    h = {h} (interval)")
        details['steps'].append(f"  ")
        details['steps'].append(f"  u = ({x_target} - {self.years[0]}) / {h}")
        details['steps'].append(f"  u = {(x_target - self.years[0])} / {h}")
        
        u = (x_target - self.years[0]) / h
        details['steps'].append(f"  u = {u:.6f}")
        
        # Step 4: Apply Newton's forward formula
        details['steps'].append("")
        details['steps'].append(f"Step 4: Apply Newton's Forward Interpolation Formula")
        details['steps'].append(f"  ")
        details['steps'].append(f"  f({x_target}) = f(x₀) + u·Δf(x₀) + [u(u-1)/2!]·Δ²f(x₀) + ...")
        details['steps'].append(f"  ")
        
        result = diff_table[0][0]
        details['steps'].append(f"  Term 0 (Base value):")
        details['steps'].append(f"    f(x₀) = {diff_table[0][0]:.6f}")
        details['steps'].append(f"  ")
        
        for i in range(1, n):
            u_term = self._calculate_u_forward(u, i)
            factorial_term = self._factorial(i)
            term_value = (u_term * diff_table[0][i]) / factorial_term
            result += term_value
            
            # Build u expression
            u_parts = ["u"]
            for k in range(1, i):
                u_parts.append(f"(u-{k})")
            u_expr = "·".join(u_parts)
            
            details['steps'].append(f"  Term {i}:")
            details['steps'].append(f"    [{u_expr}/{i}!] × Δ^{i}f(x₀)")
            details['steps'].append(f"    = [{u_term:.6f}/{factorial_term}] × {diff_table[0][i]:.6f}")
            details['steps'].append(f"    = {term_value:.6f}")
            details['steps'].append(f"  ")
        
        details['steps'].append("")
        details['steps'].append(f"Step 5: Sum all terms")
        sum_expr = f"{diff_table[0][0]:.6f}"
        for i in range(1, n):
            u_term = self._calculate_u_forward(u, i)
            factorial_term = self._factorial(i)
            term_value = (u_term * diff_table[0][i]) / factorial_term
            sum_expr += f" + {term_value:.6f}"
        details['steps'].append(f"  f({x_target}) = {sum_expr}")
        details['steps'].append(f"  f({x_target}) = {result:.6f}%")
        
        return result, details
    
    def _newton_backward_interpolation(self, x_target: int) -> Tuple[float, Dict]:
        """
        Newton's Backward Interpolation with detailed steps
        """
        n = len(self.years)
        h = self.h
        
        details = {
            'method': 'Newton Backward Interpolation',
            'formula': 'f(a + nh + uh) = f(a+nh) + u∇f(a+nh) + [u(u+1)/2!]∇²f(a+nh) + [u(u+1)(u+2)/3!]∇³f(a+nh) + ...',
            'steps': []
        }
        
        # Step 1: Create backward difference table
        diff_table = [[0] * n for _ in range(n)]
        for i in range(n):
            diff_table[i][0] = self.rates[i]
        
        details['steps'].append(f"Step 1: Given data points")
        for i in range(n):
            details['steps'].append(f"  x_{i} = {self.years[i]}, f(x_{i}) = {self.rates[i]:.4f}")
        
        # Calculate backward differences
        details['steps'].append(f"\nStep 2: Calculate backward differences")
        details['steps'].append(f"  ∇f(x_i) = f(x_i) - f(x_(i-1))")
        
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                diff_table[i][j] = diff_table[i][j-1] - diff_table[i-1][j-1]
                if j == 1:
                    details['steps'].append(f"  ∇f(x_{i}) = {diff_table[i][j-1]:.6f} - {diff_table[i-1][j-1]:.6f} = {diff_table[i][j]:.6f}")
        
        # Add difference table
        details['difference_table'] = self._format_difference_table(diff_table, "Backward")
        
        # Step 3: Calculate u
        u = (x_target - self.years[n-1]) / h
        details['steps'].append(f"\nStep 3: Calculate u")
        details['steps'].append(f"  u = (x - x_n) / h")
        details['steps'].append(f"  u = ({x_target} - {self.years[n-1]}) / {h}")
        details['steps'].append(f"  u = {u:.6f}")
        
        # Step 4: Apply Newton's backward formula
        details['steps'].append(f"\nStep 4: Apply Newton's Backward Interpolation Formula")
        details['steps'].append(f"  f({x_target}) = f(x_n)")
        
        result = diff_table[n-1][0]
        details['steps'].append(f"         = {diff_table[n-1][0]:.6f}")
        
        for i in range(1, n):
            u_term = self._calculate_u_backward(u, i)
            factorial_term = self._factorial(i)
            term_value = (u_term * diff_table[n-1][i]) / factorial_term
            result += term_value
            
            # Build u expression
            u_expr = "u"
            for k in range(1, i):
                u_expr += f"(u+{k})"
            
            details['steps'].append(f"         + [{u_expr}/{i}!] × ∇^{i}f(x_n)")
            details['steps'].append(f"         + [{u_term:.6f}/{factorial_term}] × {diff_table[n-1][i]:.6f}")
            details['steps'].append(f"         + {term_value:.6f}")
        
        details['steps'].append(f"\nStep 5: Final Result")
        details['steps'].append(f"  f({x_target}) ≈ {result:.6f}%")
        
        return result, details
    
    def _lagrange_interpolation(self, x_target: int) -> Tuple[float, Dict]:
        """
        Lagrange Interpolation with detailed steps
        """
        n = len(self.years)
        
        details = {
            'method': 'Lagrange Interpolation',
            'formula': 'f(x) = Σ[i=0 to n] y_i × L_i(x), where L_i(x) = Π[j≠i] (x-x_j)/(x_i-x_j)',
            'steps': []
        }
        
        details['steps'].append(f"Step 1: Given data points")
        for i in range(n):
            details['steps'].append(f"  x_{i} = {self.years[i]}, f(x_{i}) = {self.rates[i]:.4f}")
        
        details['steps'].append(f"\nStep 2: Calculate Lagrange basis polynomials L_i({x_target})")
        
        result = 0.0
        lagrange_terms = []
        
        for i in range(n):
            details['steps'].append(f"\n  For i = {i}:")
            
            # Calculate L_i(x)
            numerator_parts = []
            denominator_parts = []
            L_i = 1.0
            
            for j in range(n):
                if i != j:
                    numerator_parts.append(f"({x_target}-{self.years[j]})")
                    denominator_parts.append(f"({self.years[i]}-{self.years[j]})")
                    L_i *= (x_target - self.years[j]) / (self.years[i] - self.years[j])
            
            numerator_expr = " × ".join(numerator_parts)
            denominator_expr = " × ".join(denominator_parts)
            
            details['steps'].append(f"    L_{i}({x_target}) = {numerator_expr} / {denominator_expr}")
            
            # Calculate numerical values
            num_val = 1.0
            den_val = 1.0
            for j in range(n):
                if i != j:
                    num_val *= (x_target - self.years[j])
                    den_val *= (self.years[i] - self.years[j])
            
            details['steps'].append(f"    L_{i}({x_target}) = {num_val:.6f} / {den_val:.6f} = {L_i:.6f}")
            
            term_value = self.rates[i] * L_i
            lagrange_terms.append(term_value)
            result += term_value
            
            details['steps'].append(f"    Term_{i} = f(x_{i}) × L_{i}({x_target}) = {self.rates[i]:.4f} × {L_i:.6f} = {term_value:.6f}")
        
        details['steps'].append(f"\nStep 3: Sum all terms")
        details['steps'].append(f"  f({x_target}) = " + " + ".join([f"{term:.6f}" for term in lagrange_terms]))
        details['steps'].append(f"  f({x_target}) ≈ {result:.6f}%")
        
        return result, details
    
    def interpolate_rate(self, year: int) -> Tuple[float, str, Dict]:
        """
        Interpolate the rate for a given year with detailed steps
        Returns: (interpolated_rate, method_used, calculation_details)
        """
        # If year is in historical data, return actual rate
        if year in self.years:
            idx = self.years.index(year)
            details = {
                'method': 'Actual Data',
                'steps': [f"Year {year} found in historical data: {self.rates[idx]:.4f}%"]
            }
            return self.rates[idx], "Actual Data", details
        
        # Choose interpolation method based on position of THIS specific year
        if self.has_equal_intervals:
            # Calculate midpoint of data range
            data_start = min(self.years)
            data_end = max(self.years)
            mid_point = (data_start + data_end) / 2
            
            # Decide method based on which half of data range this year falls in
            if year <= mid_point:
                rate, details = self._newton_forward_interpolation(year)
                method = "Newton Forward Interpolation"
            else:
                rate, details = self._newton_backward_interpolation(year)
                method = "Newton Backward Interpolation"
        else:
            rate, details = self._lagrange_interpolation(year)
            method = "Lagrange Interpolation"
        
        # Apply risk factor for volatility
        original_rate = rate
        if self.risk_factor > 0:
            volatility = np.random.normal(0, self.risk_factor * rate * 0.1)
            rate += volatility
            details['steps'].append(f"\nStep 6: Apply Risk Factor (volatility = {self.risk_factor})")
            details['steps'].append(f"  Random adjustment: {volatility:+.6f}%")
            details['steps'].append(f"  Final rate: {rate:.6f}%")
        
        return rate, method, details
    
    def calculate_growth(self) -> Dict:
        """
        Calculate year-by-year investment growth with detailed interpolation steps
        """
        # Determine actual methods that will be used
        data_start = min(self.years)
        data_end = max(self.years)
        mid_point = (data_start + data_end) / 2
        
        # Check which method will actually be used
        if self.has_equal_intervals:
            if self.target_year <= mid_point:
                actual_method = 'Newton Forward'
            else:
                actual_method = 'Newton Backward'
        else:
            actual_method = 'Lagrange'
        
        results = {
            'yearly_data': [],
            'interpolation_method': actual_method,
            'method_reason': self._get_method_reason(),
            'summary': {},
            'calculation_steps': [],
            'interpolation_details': []  # NEW: detailed interpolation calculations
        }
        
        current_value = self.initial_investment
        total_contributions = self.initial_investment
        
        results['calculation_steps'].append("="*70)
        results['calculation_steps'].append("INVESTMENT GROWTH CALCULATION")
        results['calculation_steps'].append("="*70)
        results['calculation_steps'].append(f"Initial Investment: ${self.initial_investment:,.2f}")
        results['calculation_steps'].append(f"Investment Period: {self.start_year} to {self.target_year}")
        results['calculation_steps'].append(f"Interpolation Method: {self.interpolation_method}")
        results['calculation_steps'].append("="*70)
        
        # Calculate for each year from start to target
        for year in range(self.start_year, self.target_year + 1):
            # Get contribution for this year
            if year in self.variable_contributions:
                contribution = self.variable_contributions[year]
            elif year == self.start_year:
                contribution = 0  # Initial investment already added
            else:
                if self.contribution_frequency == 'monthly':
                    contribution = self.recurring_contribution * 12
                else:
                    contribution = self.recurring_contribution
            
            # Add contribution at beginning of year
            current_value += contribution
            total_contributions += contribution
            
            # Interpolate rate for this year with detailed steps
            rate, method, interp_details = self.interpolate_rate(year)
            
            # Store detailed interpolation steps
            results['interpolation_details'].append({
                'year': year,
                'details': interp_details
            })
            
            # Calculate growth for this year
            growth = current_value * (rate / 100)
            current_value += growth
            
            # Store yearly data
            yearly_record = {
                'year': year,
                'rate': round(rate, 4),
                'contribution': round(contribution, 2),
                'growth': round(growth, 2),
                'value': round(current_value, 2),
                'interpolation_method': method
            }
            results['yearly_data'].append(yearly_record)
            
            # Add calculation step
            results['calculation_steps'].append(f"\n--- Year {year} ---")
            results['calculation_steps'].append(f"Rate (interpolated): {rate:.4f}%")
            results['calculation_steps'].append(f"Contribution: ${contribution:,.2f}")
            results['calculation_steps'].append(f"Balance after contribution: ${current_value - growth:,.2f}")
            results['calculation_steps'].append(f"Growth this year: ${growth:,.2f}")
            results['calculation_steps'].append(f"Ending balance: ${current_value:,.2f}")
        
        # Calculate summary metrics
        final_value = current_value
        total_growth = final_value - total_contributions
        years_count = self.target_year - self.start_year + 1
        avg_annual_growth = (total_growth / total_contributions / years_count) * 100 if total_contributions > 0 else 0
        
        yearly_growths = [item['growth'] for item in results['yearly_data'] if item['growth'] > 0]
        max_growth = max(yearly_growths) if yearly_growths else 0
        min_growth = min(yearly_growths) if yearly_growths else 0
        
        results['summary'] = {
            'initial_investment': round(self.initial_investment, 2),
            'total_contributions': round(total_contributions, 2),
            'final_value': round(final_value, 2),
            'total_growth': round(total_growth, 2),
            'avg_annual_growth_rate': round(avg_annual_growth, 2),
            'max_yearly_growth': round(max_growth, 2),
            'min_yearly_growth': round(min_growth, 2),
            'years': years_count
        }
        
        results['calculation_steps'].append("\n" + "="*70)
        results['calculation_steps'].append("SUMMARY")
        results['calculation_steps'].append("="*70)
        results['calculation_steps'].append(f"Total Contributions: ${total_contributions:,.2f}")
        results['calculation_steps'].append(f"Final Value: ${final_value:,.2f}")
        results['calculation_steps'].append(f"Total Growth: ${total_growth:,.2f}")
        results['calculation_steps'].append(f"Average Annual Growth Rate: {avg_annual_growth:.2f}%")
        
        return results
    
    def _get_method_reason(self) -> str:
        """Get explanation for why interpolation methods are chosen"""
        if self.has_equal_intervals:
            data_start = min(self.years)
            data_end = max(self.years)
            mid_point = (data_start + data_end) / 2
            
            if self.target_year <= mid_point:
                return (f"Newton Forward Interpolation is used because the data has equal intervals (h = {self.h} years) "
                       f"and the target year ({self.target_year}) is in the first half of the data range ({data_start} to {int(mid_point)}). "
                       f"Newton Forward is most accurate when interpolating values near the beginning of the dataset.")
            else:
                return (f"Newton Backward Interpolation is used because the data has equal intervals (h = {self.h} years) "
                       f"and the target year ({self.target_year}) is in the second half of the data range ({int(mid_point)+1} to {data_end}). "
                       f"Newton Backward is most accurate when interpolating values near the end of the dataset.")
        else:
            return "Lagrange Interpolation is used because the historical data has unequal intervals, making it the most suitable method for accurate interpolation across the entire range."