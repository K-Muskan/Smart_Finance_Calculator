import math
from typing import List, Dict, Tuple, Optional
import sys 

class StockTrendAnalyzer:
    def __init__(self, dates: List[str], prices: List[float], company_name: str = "Company"):
        self.dates = dates
        self.prices = prices
        self.company_name = company_name
        self.n = len(prices)
        
        
        self.x_values = list(range(len(dates)))
        self.y_values = prices
        
        
        self.current_coefficients = None
        self.current_model_type = None
        
        
        if self.n < 2:
            raise ValueError("At least 2 data points required for analysis")
        
        if len(dates) != len(prices):
            raise ValueError("Dates and prices must have same length")
    
    def calculate_statistics(self) -> Dict:
       
        mean_price = sum(self.prices) / self.n

        variance = sum((p - mean_price) ** 2 for p in self.prices) / self.n
        std_dev = math.sqrt(variance)

        min_price = min(self.prices)
        max_price = max(self.prices)

        price_change = self.prices[-1] - self.prices[0]
        price_change_percent = (price_change / self.prices[0]) * 100 if self.prices[0] != 0 else 0
        
        return {
            'mean': round(mean_price, 4),
            'std_dev': round(std_dev, 4),
            'min': round(min_price, 4),
            'max': round(max_price, 4),
            'price_change': round(price_change, 4),
            'price_change_percent': round(price_change_percent, 2),
            'data_points': self.n
        }
    
    def linear_curve_fitting(self) -> Dict:
        
        x = self.x_values
        y = self.y_values
        n = self.n
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x_squared = sum(x_i ** 2 for x_i in x)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x_squared - sum_x ** 2
        
        if denominator == 0:
            raise ValueError("Cannot perform linear regression: denominator is zero")
        
        m = numerator / denominator
        c = (sum_y - m * sum_x) / n
        
        self.current_coefficients = [c, m]  
        self.current_model_type = 'linear'
        
        y_mean = sum_y / n
        ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
        ss_residual = sum((y[i] - (m * x[i] + c)) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        predictions = [m * x_i + c for x_i in x]

        residuals = [y[i] - predictions[i] for i in range(n)]

        future_x = list(range(n, n + 30))
        future_predictions = [m * x_i + c for x_i in future_x]
        
        return {
            'method': 'Linear Curve Fitting',
            'equation': f'y = {m:.6f}x + {c:.6f}',
            'coefficients': {
                'slope': round(m, 6),
                'intercept': round(c, 6)
            },
            'r_squared': round(r_squared, 6),
            'accuracy_percent': round(r_squared * 100, 2),
            'predictions': [round(p, 4) for p in predictions],
            'residuals': [round(r, 4) for r in residuals],
            'future_predictions': [round(p, 4) for p in future_predictions],
            'trend': 'Upward' if m > 0 else 'Downward' if m < 0 else 'Flat',
            'calculation_steps': {
                'sum_x': round(sum_x, 4),
                'sum_y': round(sum_y, 4),
                'sum_xy': round(sum_xy, 4),
                'sum_x_squared': round(sum_x_squared, 4),
                'numerator': round(numerator, 4),
                'denominator': round(denominator, 4)
            }
        }
    
    def polynomial_curve_fitting(self, degree: int = 2) -> Dict:
        
        if degree not in [2, 3]:
            raise ValueError("Only degree 2 (quadratic) or 3 (cubic) supported")
        
        x = self.x_values
        y = self.y_values
        n = self.n

        size = degree + 1

        matrix = [[0.0] * (size + 1) for _ in range(size)]

        for i in range(size):
            for j in range(size):
                power = i + j
                matrix[i][j] = sum(x_val ** power for x_val in x)
            
            matrix[i][size] = sum(x[k] ** i * y[k] for k in range(n))
        
        coefficients = self._gaussian_elimination(matrix)
        
        self.current_coefficients = coefficients
        self.current_model_type = 'quadratic' if degree == 2 else 'cubic'

        if degree == 2:
            equation = f'y = {coefficients[2]:.6f}x² + {coefficients[1]:.6f}x + {coefficients[0]:.6f}'
            method_name = 'Quadratic Curve Fitting'
        else:  
            equation = f'y = {coefficients[3]:.6f}x³ + {coefficients[2]:.6f}x² + {coefficients[1]:.6f}x + {coefficients[0]:.6f}'
            method_name = 'Cubic Curve Fitting'

        predictions = [self._polynomial_predict(x_i, coefficients) for x_i in x]

        y_mean = sum(y) / n
        ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
        ss_residual = sum((y[i] - predictions[i]) ** 2 for i in range(n))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        residuals = [y[i] - predictions[i] for i in range(n)]

        future_x = list(range(n, n + 30))
        future_predictions = [self._polynomial_predict(x_i, coefficients) for x_i in future_x]

        last_x = n - 1
        if degree == 2:
            derivative = 2 * coefficients[2] * last_x + coefficients[1]
        else:  # degree == 3
            derivative = 3 * coefficients[3] * last_x**2 + 2 * coefficients[2] * last_x + coefficients[1]
        
        trend = 'Upward' if derivative > 0 else 'Downward' if derivative < 0 else 'Flat'
        
        return {
            'method': method_name,
            'degree': degree,
            'equation': equation,
            'coefficients': {f'c{i}': round(coefficients[i], 6) for i in range(len(coefficients))},
            'r_squared': round(r_squared, 6),
            'accuracy_percent': round(r_squared * 100, 2),
            'predictions': [round(p, 4) for p in predictions],
            'residuals': [round(r, 4) for r in residuals],
            'future_predictions': [round(p, 4) for p in future_predictions],
            'trend': trend,
            'derivative_at_end': round(derivative, 6)
        }
    
    def _gaussian_elimination(self, matrix: List[List[float]]) -> List[float]:

        n = len(matrix)
        
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k

            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

            for k in range(i + 1, n):
                if matrix[i][i] != 0:
                    factor = matrix[k][i] / matrix[i][i]
                    for j in range(i, n + 1):
                        matrix[k][j] -= factor * matrix[i][j]

        solution = [0.0] * n
        for i in range(n - 1, -1, -1):
            solution[i] = matrix[i][n]
            for j in range(i + 1, n):
                solution[i] -= matrix[i][j] * solution[j]
            if matrix[i][i] != 0:
                solution[i] /= matrix[i][i]
        
        return solution
    
    def _polynomial_predict(self, x_val: float, coefficients: List[float]) -> float:

        result = 0.0
        for i, coef in enumerate(coefficients):
            result += coef * (x_val ** i)
        return result
    
    def _evaluate_polynomial(self, x: float) -> float:

        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        return self._polynomial_predict(x, self.current_coefficients)
    
    def _evaluate_polynomial_derivative(self, x: float) -> float:
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        coeffs = self.current_coefficients
        result = 0.0

        for i in range(1, len(coeffs)):
            result += i * coeffs[i] * (x ** (i - 1))
        
        return result
    
    # ======================== ROOT FINDING METHODS ========================
    
    def bisection_method(self, target_price: float, max_iterations: int = 100, tolerance: float = 0.0001) -> Dict:
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        def f(x):
            return self._evaluate_polynomial(x) - target_price

        a = 0.0
        b = float(self.n * 3)  # Look up to 3x the current data range
        

        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            # Root might not exist in this interval
            # Try to adjust bounds
            if fa > 0:  # Target is below both endpoints
                return {
                    'method': 'Bisection Method',
                    'success': False,
                    'message': f'Target price ${target_price:.2f} is below the predicted range',
                    'iterations': 0,
                    'current_price': round(self.prices[-1], 4),
                    'target_price': round(target_price, 4)
                }
            else:  # Target is above both endpoints
                b = self.n * 10  # Extend search range
                fb = f(b)
                if fa * fb > 0:
                    return {
                        'method': 'Bisection Method',
                        'success': False,
                        'message': f'Target price ${target_price:.2f} is too far in the future to predict accurately',
                        'iterations': 0,
                        'current_price': round(self.prices[-1], 4),
                        'target_price': round(target_price, 4)
                    }
        
        # Bisection algorithm
        iteration = 0
        error_history = []
        
        while iteration < max_iterations:
            # Calculate midpoint
            c = (a + b) / 2.0
            fc = f(c)
            
            # Calculate error
            error = abs(fc)
            error_history.append(error)
            
            # Check if we found the root
            if error < tolerance or abs(b - a) < tolerance:
                days_from_now = c - (self.n - 1)
                predicted_date_index = int(round(c))
                
                return {
                    'method': 'Bisection Method',
                    'success': True,
                    'target_price': round(target_price, 4),
                    'root_x': round(c, 4),
                    'days_from_last_data': round(days_from_now, 2),
                    'predicted_price': round(self._evaluate_polynomial(c), 4),
                    'error': round(error, 6),
                    'iterations': iteration + 1,
                    'tolerance_used': tolerance,
                    'error_history': [round(e, 6) for e in error_history],
                    'convergence_rate': 'Linear (Guaranteed)',
                    'message': f'Stock will reach ${target_price:.2f} in approximately {days_from_now:.1f} days'
                }
            
            # Update interval
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            iteration += 1
        
        # Maximum iterations reached
        return {
            'method': 'Bisection Method',
            'success': False,
            'message': f'Maximum iterations ({max_iterations}) reached without convergence',
            'iterations': max_iterations,
            'target_price': round(target_price, 4),
            'best_approximation': round(c, 4),
            'error': round(error, 6)
        }
    
    def false_position_method(self, target_price: float, max_iterations: int = 100, tolerance: float = 0.0001) -> Dict:
        """
        Find when the stock price will reach a target price using False Position Method (Regula Falsi)
        Usually converges faster than bisection
        
        Args:
            target_price: The target price to find
            max_iterations: Maximum number of iterations
            tolerance: Acceptable error tolerance
            
        Returns:
            Dictionary with root finding results
        """
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        # Define the function: f(x) - target_price
        def f(x):
            return self._evaluate_polynomial(x) - target_price
        
        # Find search interval
        a = 0.0
        b = float(self.n * 3)
        
        fa = f(a)
        fb = f(b)
        
        # Check if root exists in interval
        if fa * fb > 0:
            if fa > 0:
                return {
                    'method': 'False Position Method',
                    'success': False,
                    'message': f'Target price ${target_price:.2f} is below the predicted range',
                    'iterations': 0,
                    'current_price': round(self.prices[-1], 4),
                    'target_price': round(target_price, 4)
                }
            else:
                b = self.n * 10
                fb = f(b)
                if fa * fb > 0:
                    return {
                        'method': 'False Position Method',
                        'success': False,
                        'message': f'Target price ${target_price:.2f} is too far in the future',
                        'iterations': 0,
                        'current_price': round(self.prices[-1], 4),
                        'target_price': round(target_price, 4)
                    }
        
        # False Position algorithm
        iteration = 0
        error_history = []
        c_old = a
        
        while iteration < max_iterations:
            # Calculate next approximation using false position formula
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            
            # Calculate error
            error = abs(fc)
            error_history.append(error)
            
            # Check convergence
            if error < tolerance or abs(c - c_old) < tolerance:
                days_from_now = c - (self.n - 1)
                
                return {
                    'method': 'False Position Method',
                    'success': True,
                    'target_price': round(target_price, 4),
                    'root_x': round(c, 4),
                    'days_from_last_data': round(days_from_now, 2),
                    'predicted_price': round(self._evaluate_polynomial(c), 4),
                    'error': round(error, 6),
                    'iterations': iteration + 1,
                    'tolerance_used': tolerance,
                    'error_history': [round(e, 6) for e in error_history],
                    'convergence_rate': 'Super-linear',
                    'message': f'Stock will reach ${target_price:.2f} in approximately {days_from_now:.1f} days'
                }
            
            # Update interval
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            c_old = c
            iteration += 1
        
        # Maximum iterations reached
        return {
            'method': 'False Position Method',
            'success': False,
            'message': f'Maximum iterations ({max_iterations}) reached',
            'iterations': max_iterations,
            'target_price': round(target_price, 4),
            'best_approximation': round(c, 4),
            'error': round(error, 6)
        }
    
    def newton_raphson_method(self, target_price: float, max_iterations: int = 100, tolerance: float = 0.0001) -> Dict:
       
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        # Define the function and its derivative
        def f(x):
            return self._evaluate_polynomial(x) - target_price
        
        def f_prime(x):
            return self._evaluate_polynomial_derivative(x)
        
        # Initial guess: start from the end of the data
        x = float(self.n - 1)
        
        # Check if we should look forward or backward
        current_value = self._evaluate_polynomial(x)
        if target_price > current_value:
            x = float(self.n + 10)  # Look forward
        elif target_price < self.prices[0]:
            x = -10.0  # Look backward (if makes sense)
        
        iteration = 0
        error_history = []
        
        while iteration < max_iterations:
            fx = f(x)
            fpx = f_prime(x)
            
            # Check for zero derivative (would cause division by zero)
            if abs(fpx) < 1e-10:
                return {
                    'method': 'Newton-Raphson Method',
                    'success': False,
                    'message': 'Derivative too close to zero, method failed',
                    'iterations': iteration,
                    'target_price': round(target_price, 4)
                }
            
            # Calculate error
            error = abs(fx)
            error_history.append(error)
            
            # Check convergence
            if error < tolerance:
                days_from_now = x - (self.n - 1)
                
                # Additional validation
                if x < -365:  # More than a year in the past
                    return {
                        'method': 'Newton-Raphson Method',
                        'success': False,
                        'message': f'Target price ${target_price:.2f} is in the historical past',
                        'iterations': iteration + 1,
                        'target_price': round(target_price, 4)
                    }
                
                return {
                    'method': 'Newton-Raphson Method',
                    'success': True,
                    'target_price': round(target_price, 4),
                    'root_x': round(x, 4),
                    'days_from_last_data': round(days_from_now, 2),
                    'predicted_price': round(self._evaluate_polynomial(x), 4),
                    'error': round(error, 6),
                    'iterations': iteration + 1,
                    'tolerance_used': tolerance,
                    'error_history': [round(e, 6) for e in error_history],
                    'convergence_rate': 'Quadratic (Very Fast)',
                    'message': f'Stock will reach ${target_price:.2f} in approximately {days_from_now:.1f} days'
                }
            
            # Newton-Raphson update
            x_new = x - fx / fpx
            
            # Check for divergence
            if abs(x_new) > self.n * 1000:
                return {
                    'method': 'Newton-Raphson Method',
                    'success': False,
                    'message': 'Method diverged, target may be unreachable',
                    'iterations': iteration + 1,
                    'target_price': round(target_price, 4)
                }
            
            x = x_new
            iteration += 1
        
        # Maximum iterations reached
        return {
            'method': 'Newton-Raphson Method',
            'success': False,
            'message': f'Maximum iterations ({max_iterations}) reached',
            'iterations': max_iterations,
            'target_price': round(target_price, 4),
            'best_approximation': round(x, 4),
            'error': round(error, 6)
        }
    
    def compare_root_finding_methods(self, target_price: float) -> Dict:
        
        # Run all methods
        bisection = self.bisection_method(target_price)
        false_position = self.false_position_method(target_price)
        newton_raphson = self.newton_raphson_method(target_price)
        
        successful_methods = []
        
        if bisection['success']:
            successful_methods.append(bisection)
        if false_position['success']:
            successful_methods.append(false_position)
        if newton_raphson['success']:
            successful_methods.append(newton_raphson)
        
        # Find fastest method (least iterations)
        if successful_methods:
            fastest = min(successful_methods, key=lambda x: x['iterations'])
        else:
            fastest = None
        
        return {
            'target_price': round(target_price, 4),
            'bisection': bisection,
            'false_position': false_position,
            'newton_raphson': newton_raphson,
            'fastest_method': fastest['method'] if fastest else None,
            'all_successful': bisection['success'] and false_position['success'] and newton_raphson['success']
        }
    
    # ======================== NUMERICAL ERROR ANALYSIS ========================
    
    def calculate_truncation_error(self) -> Dict:
        
        if self.current_model_type is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        # Get predictions from current model
        current_predictions = [self._polynomial_predict(x, self.current_coefficients) 
                              for x in self.x_values]
        
        # Fit a higher degree model for comparison
        if self.current_model_type == 'linear':
            higher_degree = 2
            comparison_model = self.polynomial_curve_fitting(degree=2)
        elif self.current_model_type == 'quadratic':
            higher_degree = 3
            comparison_model = self.polynomial_curve_fitting(degree=3)
        else:  # cubic
            # For cubic, we'll use the residuals as estimate
            higher_degree = 3
            residuals = [self.y_values[i] - current_predictions[i] for i in range(self.n)]
            max_error = max(abs(r) for r in residuals)
            mean_error = sum(abs(r) for r in residuals) / self.n
            
            return {
                'current_model': self.current_model_type.title(),
                'truncation_error_type': 'Residual-based (highest degree)',
                'max_truncation_error': round(max_error, 6),
                'mean_truncation_error': round(mean_error, 6),
                'error_percentage': round((mean_error / sum(self.y_values) * self.n) * 100, 4),
                'message': 'Cubic is highest degree supported. Error estimated from residuals.'
            }
        
        higher_predictions = comparison_model['predictions']
        truncation_errors = [abs(higher_predictions[i] - current_predictions[i]) 
                           for i in range(self.n)]
        
        max_error = max(truncation_errors)
        mean_error = sum(truncation_errors) / self.n
        
        return {
            'current_model': self.current_model_type.title(),
            'comparison_model': f'Degree {higher_degree}',
            'max_truncation_error': round(max_error, 6),
            'mean_truncation_error': round(mean_error, 6),
            'error_percentage': round((mean_error / sum(self.y_values) * self.n) * 100, 4),
            'truncation_errors': [round(e, 6) for e in truncation_errors],
            'message': f'Truncation error estimated by comparing with degree {higher_degree} polynomial'
        }
    
    def calculate_roundoff_error(self) -> Dict:
        
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        # Machine epsilon (smallest number where 1.0 + epsilon != 1.0)
        machine_epsilon = sys.float_info.epsilon
        
        # Estimate accumulated round-off error in polynomial evaluation
        # For polynomial of degree n, error accumulates as: O(n * epsilon * |x|)
        degree = len(self.current_coefficients) - 1
        
        max_x = max(abs(x) for x in self.x_values)
        estimated_roundoff = degree * machine_epsilon * max_x
        
        # Calculate actual numerical stability by re-computing with slightly perturbed data
        perturbed_predictions = []
        original_predictions = []
        
        for x in self.x_values:
            # Original calculation
            original = self._polynomial_predict(x, self.current_coefficients)
            original_predictions.append(original)
            
            # Perturbed calculation (simulate round-off)
            perturbed_coeffs = [c * (1 + machine_epsilon) for c in self.current_coefficients]
            perturbed = self._polynomial_predict(x, perturbed_coeffs)
            perturbed_predictions.append(perturbed)
        
        # Calculate actual error
        actual_errors = [abs(perturbed_predictions[i] - original_predictions[i]) 
                        for i in range(self.n)]
        max_actual_error = max(actual_errors)
        mean_actual_error = sum(actual_errors) / self.n
        
        return {
            'machine_epsilon': f'{machine_epsilon:.2e}',
            'polynomial_degree': degree,
            'estimated_roundoff_error': f'{estimated_roundoff:.2e}',
            'max_actual_error': f'{max_actual_error:.2e}',
            'mean_actual_error': f'{mean_actual_error:.2e}',
            'relative_error_percentage': round((mean_actual_error / sum(abs(p) for p in original_predictions) * self.n) * 100, 8),
            'message': 'Round-off errors are negligible for this calculation',
            'stability': 'Excellent' if max_actual_error < 1e-6 else 'Good' if max_actual_error < 1e-4 else 'Moderate'
        }
    
    def calculate_condition_number(self) -> Dict:
        
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        # Build the Vandermonde matrix used in polynomial fitting
        n = len(self.current_coefficients)
        matrix = []
        
        for x in self.x_values:
            row = [x ** i for i in range(n)]
            matrix.append(row)
        
        # Calculate condition number using Frobenius norm
        # cond(A) ≈ ||A|| * ||A^-1||
        
        # Calculate matrix norm (Frobenius norm)
        norm_A = math.sqrt(sum(sum(val ** 2 for val in row) for row in matrix))
        
        # Estimate inverse norm using the determinant method (simplified)
        # For a more accurate condition number, we'd need full matrix inversion
        # Here we provide an estimate based on the range of x values
        x_range = max(self.x_values) - min(self.x_values)
        estimated_cond = (x_range ** (n - 1)) / max(1, min(self.x_values))
        
        # Classify stability
        if estimated_cond < 10:
            stability = 'Excellent - Well-conditioned'
            risk = 'Very Low'
        elif estimated_cond < 100:
            stability = 'Good - Moderately conditioned'
            risk = 'Low'
        elif estimated_cond < 1000:
            stability = 'Fair - Somewhat ill-conditioned'
            risk = 'Moderate'
        else:
            stability = 'Poor - Ill-conditioned'
            risk = 'High'
        
        return {
            'estimated_condition_number': round(estimated_cond, 4),
            'matrix_norm': round(norm_A, 4),
            'stability_assessment': stability,
            'numerical_risk': risk,
            'degree': n - 1,
            'data_range': round(x_range, 2),
            'message': f'Condition number of {estimated_cond:.2f} indicates {stability.lower()}',
            'recommendation': 'Current model is stable' if estimated_cond < 100 else 
                            'Consider using lower degree polynomial for better stability'
        }
    
    def comprehensive_error_analysis(self) -> Dict:
        """
        Perform comprehensive numerical error analysis
        
        Returns:
            Dictionary with all error analysis results
        """
        truncation = self.calculate_truncation_error()
        roundoff = self.calculate_roundoff_error()
        condition = self.calculate_condition_number()
        
        # Overall assessment
        overall_quality = 'Excellent'
        if condition['numerical_risk'] != 'Very Low' and condition['numerical_risk'] != 'Low':
            overall_quality = 'Good'
        if truncation.get('error_percentage', 0) > 5:
            overall_quality = 'Moderate'
        
        return {
            'truncation_error': truncation,
            'roundoff_error': roundoff,
            'condition_number': condition,
            'overall_quality': overall_quality,
            'summary': {
                'model_type': self.current_model_type.title() if self.current_model_type else 'None',
                'truncation_error_mean': truncation.get('mean_truncation_error', 0),
                'roundoff_error_mean': roundoff['mean_actual_error'],
                'condition_number': condition['estimated_condition_number'],
                'stability': condition['stability_assessment']
            }
        }
    
    # ======================== EXISTING METHODS ========================
    
    def compare_models(self) -> Dict:
        """
        Compare all curve fitting models and recommend the best one
        
        Returns:
            Dictionary with comparison results
        """
        # Run all models
        linear = self.linear_curve_fitting()
        quadratic = self.polynomial_curve_fitting(degree=2)
        cubic = self.polynomial_curve_fitting(degree=3)
        
        models = [
            {'name': 'Linear', 'r_squared': linear['r_squared'], 'data': linear},
            {'name': 'Quadratic', 'r_squared': quadratic['r_squared'], 'data': quadratic},
            {'name': 'Cubic', 'r_squared': cubic['r_squared'], 'data': cubic}
        ]
        
        # Find best model (highest R²)
        best_model = max(models, key=lambda x: x['r_squared'])
        
        return {
            'models': models,
            'best_model': best_model['name'],
            'best_r_squared': round(best_model['r_squared'], 6),
            'recommendation': f"The {best_model['name']} model provides the best fit with R² = {best_model['r_squared']:.4f}",
            'statistics': self.calculate_statistics()
        }
    
    def analyze_complete(self, method: str = 'auto') -> Dict:
        """
        Perform complete stock trend analysis using curve fitting
        
        Args:
            method: 'linear', 'quadratic', 'cubic', or 'auto' (best fit)
        
        Returns:
            Complete analysis results
        """
        statistics = self.calculate_statistics()
        
        if method == 'auto':
            comparison = self.compare_models()
            # Select best model
            best = comparison['best_model'].lower()
            if best == 'linear':
                curve_fitting_result = self.linear_curve_fitting()
            elif best == 'quadratic':
                curve_fitting_result = self.polynomial_curve_fitting(degree=2)
            else:  # cubic
                curve_fitting_result = self.polynomial_curve_fitting(degree=3)
            
            return {
                'company_name': self.company_name,
                'statistics': statistics,
                'curve_fitting': curve_fitting_result,
                'method_used': best.title(),
                'auto_selected': True,
                'comparison': comparison,
                'dates': self.dates,
                'prices': [round(p, 4) for p in self.prices]
            }
        else:
            # Use specified method
            if method == 'linear':
                curve_fitting_result = self.linear_curve_fitting()
            elif method == 'quadratic':
                curve_fitting_result = self.polynomial_curve_fitting(degree=2)
            elif method == 'cubic':
                curve_fitting_result = self.polynomial_curve_fitting(degree=3)
            else:
                raise ValueError(f"Invalid method: {method}")
            
            return {
                'company_name': self.company_name,
                'statistics': statistics,
                'curve_fitting': curve_fitting_result,
                'method_used': method.title(),
                'auto_selected': False,
                'dates': self.dates,
                'prices': [round(p, 4) for p in self.prices]
            }


def parse_csv_data(csv_content: str) -> Tuple[List[str], List[float]]:
    """
    Parse CSV content and extract dates and closing prices
    
    Args:
        csv_content: CSV file content as string
    
    Returns:
        Tuple of (dates, prices)
    """
    lines = csv_content.strip().split('\n')
    
    if len(lines) < 2:
        raise ValueError("CSV must contain header and at least one data row")
    
    dates = []
    prices = []
    
    # Skip header and parse data
    for line in lines[1:]:
        # Remove quotes and split
        line = line.strip().replace('"', '')
        parts = line.split(',')
        
        if len(parts) >= 5:  # Need at least: index, Date, Open, High, Low, Close
            try:
                date = parts[1]  # Date column
                close_price = float(parts[5])  # Close column
                dates.append(date)
                prices.append(close_price)
            except (ValueError, IndexError):
                continue
    
    if not dates or not prices:
        raise ValueError("No valid data found in CSV")
    
    return dates, prices
