import math
from typing import List, Dict, Tuple

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
            'trend': 'Upward' if m > 0 else 'Downward' if m < 0 else 'Flat'
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
        else:
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
    
    def newton_raphson_method(self, target_price: float, max_iterations: int = 100, tolerance: float = 0.0001) -> Dict:
        
        if self.current_coefficients is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        def f(x):
            return self._evaluate_polynomial(x) - target_price
        
        def f_prime(x):
            return self._evaluate_polynomial_derivative(x)
        
        # Initial guess based on target price
        x = float(self.n - 1)
        current_value = self._evaluate_polynomial(x)
        
        if target_price > current_value:
            x = float(self.n + 10)
        elif target_price < self.prices[0]:
            x = -10.0
        
        iteration = 0
        error_history = []
        
        while iteration < max_iterations:
            fx = f(x)
            fpx = f_prime(x)
            
            if abs(fpx) < 1e-10:
                return {
                    'method': 'Newton-Raphson Method',
                    'success': False,
                    'message': 'Derivative too close to zero, method failed',
                    'iterations': iteration,
                    'target_price': round(target_price, 4)
                }
            
            error = abs(fx)
            error_history.append(error)
            
            if error < tolerance:
                days_from_now = x - (self.n - 1)
                
                if x < -365:
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
                    'convergence_rate': 'Quadratic',
                    'message': f'Stock will reach ${target_price:.2f} in approximately {days_from_now:.1f} days'
                }
            
            x_new = x - fx / fpx
            
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
        
        return {
            'method': 'Newton-Raphson Method',
            'success': False,
            'message': f'Maximum iterations ({max_iterations}) reached',
            'iterations': max_iterations,
            'target_price': round(target_price, 4),
            'best_approximation': round(x, 4),
            'error': round(error, 6)
        }
    
    def calculate_truncation_error(self) -> Dict:
        
        if self.current_model_type is None:
            raise ValueError("No model fitted yet. Run curve fitting first.")
        
        current_predictions = [self._polynomial_predict(x, self.current_coefficients) for x in self.x_values]
        
        if self.current_model_type == 'linear':
            higher_degree = 2
            comparison_model = self.polynomial_curve_fitting(degree=2)
        elif self.current_model_type == 'quadratic':
            higher_degree = 3
            comparison_model = self.polynomial_curve_fitting(degree=3)
        else:
            # Cubic is highest, estimate from residuals
            residuals = [self.y_values[i] - current_predictions[i] for i in range(self.n)]
            max_error = max(abs(r) for r in residuals)
            mean_error = sum(abs(r) for r in residuals) / self.n
            
            return {
                'current_model': self.current_model_type.title(),
                'truncation_error_type': 'Residual-based',
                'max_truncation_error': round(max_error, 6),
                'mean_truncation_error': round(mean_error, 6),
                'error_percentage': round((mean_error / sum(self.y_values) * self.n) * 100, 4),
                'message': 'Cubic is highest degree supported. Error estimated from residuals.'
            }
        
        higher_predictions = comparison_model['predictions']
        truncation_errors = [abs(higher_predictions[i] - current_predictions[i]) for i in range(self.n)]
        
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
    
    def compare_models(self) -> Dict:
        
        linear = self.linear_curve_fitting()
        quadratic = self.polynomial_curve_fitting(degree=2)
        cubic = self.polynomial_curve_fitting(degree=3)
        
        models = [
            {'name': 'Linear', 'r_squared': linear['r_squared'], 'data': linear},
            {'name': 'Quadratic', 'r_squared': quadratic['r_squared'], 'data': quadratic},
            {'name': 'Cubic', 'r_squared': cubic['r_squared'], 'data': cubic}
        ]
        
        best_model = max(models, key=lambda x: x['r_squared'])
        
        return {
            'models': models,
            'best_model': best_model['name'],
            'best_r_squared': round(best_model['r_squared'], 6),
            'recommendation': f"The {best_model['name']} model provides the best fit with R² = {best_model['r_squared']:.4f}",
            'statistics': self.calculate_statistics()
        }
    
    def analyze_complete(self, method: str = 'auto') -> Dict:
        
        statistics = self.calculate_statistics()
        
        if method == 'auto':
            comparison = self.compare_models()
            best = comparison['best_model'].lower()
            
            if best == 'linear':
                curve_fitting_result = self.linear_curve_fitting()
            elif best == 'quadratic':
                curve_fitting_result = self.polynomial_curve_fitting(degree=2)
            else:
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

    lines = csv_content.strip().split('\n')
    
    if len(lines) < 2:
        raise ValueError("CSV must contain header and at least one data row")
    
    dates = []
    prices = []
    
    for line in lines[1:]:
        line = line.strip().replace('"', '')
        parts = line.split(',')
        
        if len(parts) >= 5:
            try:
                date = parts[1]
                close_price = float(parts[5])
                dates.append(date)
                prices.append(close_price)
            except (ValueError, IndexError):
                continue
    
    if not dates or not prices:
        raise ValueError("No valid data found in CSV")
    
    return dates, prices