from flask import Flask, render_template, request, jsonify
from modules.loan_payment import LoanCalculator
import traceback
import io
import csv
import sys
from modules.investment_growth import InvestmentGrowthCalculator
from modules.stock_trend_analyzer import StockTrendAnalyzer, parse_csv_data
from modules.savings_goal import SavingsGoalCalculator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['JSON_SORT_KEYS'] = False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/loan-analyzer')
def loan_analyzer():
    return render_template('loan-analyzer.html')

# API Endpoints
@app.route('/api/calculate', methods=['POST'])
def calculate_payment():
    """
    API endpoint to calculate loan payment using Newton-Raphson or Secant method
    
    Expected JSON input:
    {
        "principal": 50000,
        "annualRate": 5.5,
        "loanTerm": 30,
        "termUnit": "years",
        "compounding": "monthly",
        "method": "newton"  // or "secant"
    }
    
    Returns:
    {
        "monthlyPayment": 283.87,
        "totalPaid": 102193.20,
        "totalInterest": 52193.20,
        "iterations": [...],
        "numIterations": 8,
        "converged": true,
        "method": "Newton-Raphson Method",
        "monthlyRate": 0.00458,
        "totalMonths": 360
    }
    """
    try:
        data = request.get_json()
        
        required_fields = ['principal', 'annualRate', 'loanTerm']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        principal = float(data.get('principal', 0))
        annual_rate = float(data.get('annualRate', 0))
        loan_term = float(data.get('loanTerm', 0))
        term_unit = data.get('termUnit', 'years')
        compounding = data.get('compounding', 'monthly')
        method = data.get('method', 'newton')  # Default to Newton-Raphson
        
        if principal <= 0:
            return jsonify({'error': 'Principal amount must be greater than 0'}), 400
        if annual_rate < 0:
            return jsonify({'error': 'Interest rate cannot be negative'}), 400
        if loan_term <= 0:
            return jsonify({'error': 'Loan term must be greater than 0'}), 400
        
        calculator = LoanCalculator(
            principal=principal,
            annual_rate=annual_rate,
            loan_term=loan_term,
            term_unit=term_unit,
            compounding=compounding
        )
        
        # Calculate using selected method
        if method == 'secant':
            result = calculator.calculate_secant()
        else:  # newton (default)
            result = calculator.calculate_newton_raphson()
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in calculate_payment: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500

@app.route('/api/amortization', methods=['POST'])
def get_amortization():
    """
    API endpoint to generate amortization schedule
    
    Expected JSON input:
    {
        "principal": 50000,
        "monthlyPayment": 283.87,
        "monthlyRate": 0.00458,
        "totalMonths": 360
    }
    
    Returns:
    {
        "schedule": [
            {
                "month": 1,
                "payment": 283.87,
                "principal": 55.37,
                "interest": 228.50,
                "balance": 49944.63
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        principal = float(data.get('principal', 0))
        monthly_payment = float(data.get('monthlyPayment', 0))
        monthly_rate = float(data.get('monthlyRate', 0))
        total_months = int(data.get('totalMonths', 0))
        
        if principal <= 0 or monthly_payment <= 0 or total_months <= 0:
            return jsonify({'error': 'Invalid parameters for amortization'}), 400
        
        calculator = LoanCalculator(principal=principal, annual_rate=0, loan_term=0)
        schedule = calculator.generate_amortization_schedule(
            monthly_payment, monthly_rate, total_months
        )
        
        return jsonify({'schedule': schedule}), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in get_amortization: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Amortization error: {str(e)}'}), 500

@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """
    API endpoint to export amortization schedule as CSV
    
    Expected JSON input:
    {
        "schedule": [...]
    }
    
    Returns:
    {
        "csv": "month,payment,principal,interest,balance\n1,283.87,..."
    }
    """
    try:
        data = request.get_json()
        schedule = data.get('schedule', [])
        
        if not schedule:
            return jsonify({'error': 'No schedule data provided'}), 400
        
        # Generate CSV content
        output = io.StringIO()
        fieldnames = ['month', 'payment', 'principal', 'interest', 'balance']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(schedule)
        
        csv_content = output.getvalue()
        output.close()
        
        return jsonify({'csv': csv_content}), 200
    
    except Exception as e:
        print(f"Error in export_csv: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Export error: {str(e)}'}), 500
    

@app.route('/investment-growth')
def investment_growth():
    return render_template('investment-growth.html')

@app.route('/api/investment/calculate', methods=['POST'])
def calculate_investment_growth():
    """
    API endpoint to calculate investment growth using interpolation
    
    Expected JSON input:
    {
        "initialInvestment": 10000,
        "recurringContribution": 1000,
        "contributionFrequency": "yearly",
        "historicalData": [
            {"year": 2010, "rate": 5.0},
            {"year": 2012, "rate": 6.0},
            {"year": 2015, "rate": 7.0}
        ],
        "targetYear": 2025,
        "startYear": 2010,
        "variableContributions": {},
        "riskFactor": 0.1,
        "interpolationMethod": "auto"
    }
    
    Returns:
    {
        "yearly_data": [...],
        "interpolation_method": "Newton Forward",
        "method_reason": "...",
        "summary": {...},
        "calculation_steps": [...]
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        initial_investment = float(data.get('initialInvestment', 0))
        recurring_contribution = float(data.get('recurringContribution', 0))
        contribution_frequency = data.get('contributionFrequency', 'yearly')
        historical_data = data.get('historicalData', [])
        target_year = int(data.get('targetYear', 2025))
        start_year = int(data.get('startYear', 2020))
        variable_contributions = data.get('variableContributions', {})
        risk_factor = float(data.get('riskFactor', 0))
        
        # Validate inputs
        if initial_investment <= 0:
            return jsonify({'error': 'Initial investment must be greater than 0'}), 400
        
        if not historical_data or len(historical_data) < 2:
            return jsonify({'error': 'At least 2 historical data points required'}), 400
        
        if target_year <= start_year:
            return jsonify({'error': 'Target year must be greater than start year'}), 400
        
        # Parse historical data
        years = [int(item['year']) for item in historical_data]
        rates = [float(item['rate']) for item in historical_data]
        
        # Convert variable contributions to proper format
        var_contrib = {int(k): float(v) for k, v in variable_contributions.items()} if variable_contributions else {}
        
        # Create calculator instance
        calculator = InvestmentGrowthCalculator(
            initial_investment=initial_investment,
            years=years,
            rates=rates,
            target_year=target_year,
            recurring_contribution=recurring_contribution,
            contribution_frequency=contribution_frequency,
            variable_contributions=var_contrib,
            risk_factor=risk_factor
        )
        
        # Calculate growth
        result = calculator.calculate_growth()
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in calculate_investment_growth: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500

@app.route('/api/investment/parse-csv', methods=['POST'])
def parse_investment_csv():
    """
    API endpoint to parse CSV file with historical investment data
    
    Expected CSV format:
    Year,Rate(%)
    2010,5.0
    2012,6.0
    2015,7.0
    
    Returns:
    {
        "data": [
            {"year": 2010, "rate": 5.0},
            {"year": 2012, "rate": 6.0}
        ]
    }
    """
    try:
        data = request.get_json()
        csv_content = data.get('csv', '')
        
        if not csv_content:
            return jsonify({'error': 'No CSV content provided'}), 400
        
        # Parse CSV
        lines = csv_content.strip().split('\n')
        if len(lines) < 2:
            return jsonify({'error': 'CSV must contain header and at least one data row'}), 400
        
        # Skip header
        parsed_data = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    year = int(parts[0].strip())
                    rate = float(parts[1].strip())
                    parsed_data.append({'year': year, 'rate': rate})
                except (ValueError, IndexError):
                    continue
        
        if not parsed_data:
            return jsonify({'error': 'No valid data found in CSV'}), 400
        
        return jsonify({'data': parsed_data}), 200
    
    except Exception as e:
        print(f"Error in parse_investment_csv: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'CSV parsing error: {str(e)}'}), 500




@app.route('/stock-trend-analyzer')
def stock_trend_analyzer():
    return render_template('stock-trend-analyzer.html')

@app.route('/api/stock/analyze', methods=['POST'])
def analyze_stock_trend():
    try:
        data = request.get_json()
        
        company_name = data.get('companyName', 'Company')
        csv_data = data.get('csvData', '')
        method = data.get('method', 'auto')
        
        if not csv_data:
            return jsonify({'error': 'CSV data is required'}), 400
        
        if method not in ['auto', 'linear', 'quadratic', 'cubic']:
            return jsonify({'error': 'Invalid analysis method'}), 400
        
        try:
            dates, prices = parse_csv_data(csv_data)
        except ValueError as e:
            return jsonify({'error': f'CSV parsing error: {str(e)}'}), 400
        
        analyzer = StockTrendAnalyzer(
            dates=dates,
            prices=prices,
            company_name=company_name
        )
        
        result = analyzer.analyze_complete(method=method)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in analyze_stock_trend: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/api/stock/root-finding', methods=['POST'])
def stock_root_finding():
    """Single endpoint for Newton-Raphson root finding"""
    try:
        data = request.get_json()
        
        company_name = data.get('companyName', 'Company')
        dates = data.get('dates', [])
        prices = data.get('prices', [])
        method = data.get('method', 'auto')
        target_price = float(data.get('targetPrice', 0))
        
        if not dates or not prices:
            return jsonify({'error': 'Dates and prices are required'}), 400
        
        if len(dates) != len(prices):
            return jsonify({'error': 'Dates and prices must have same length'}), 400
        
        if target_price <= 0:
            return jsonify({'error': 'Target price must be greater than 0'}), 400
        
        analyzer = StockTrendAnalyzer(
            dates=dates,
            prices=prices,
            company_name=company_name
        )
        
        # Fit the model first
        analyzer.analyze_complete(method=method)
        
        # Find root using Newton-Raphson
        result = analyzer.newton_raphson_method(target_price)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in stock_root_finding: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Root finding error: {str(e)}'}), 500

@app.route('/api/stock/truncation-error', methods=['POST'])
def stock_truncation_error():
    """Calculate truncation error for the fitted model"""
    try:
        data = request.get_json()
        
        company_name = data.get('companyName', 'Company')
        dates = data.get('dates', [])
        prices = data.get('prices', [])
        method = data.get('method', 'auto')
        
        if not dates or not prices:
            return jsonify({'error': 'Dates and prices are required'}), 400
        
        analyzer = StockTrendAnalyzer(
            dates=dates, 
            prices=prices, 
            company_name=company_name
        )
        
        # Fit the model
        analyzer.analyze_complete(method=method)
        
        # Calculate truncation error
        result = analyzer.calculate_truncation_error()
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error in stock_truncation_error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Truncation error calculation failed: {str(e)}'}), 500
 
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'features': [
            'Loan Payment Calculator',
            'Investment Growth Calculator',
            'Stock Trend Analyzer',
            'Root Finding Methods',
            'Error Analysis'
        ]
    }), 200


@app.route('/api/methods-info', methods=['GET'])
def methods_info():
    """
    Get information about available numerical methods
    """
    return jsonify({
        'root_finding_methods': {
            'bisection': {
                'name': 'Bisection Method',
                'convergence': 'Linear',
                'speed': 'Slow',
                'reliability': 'Guaranteed',
                'description': 'Most stable method, guaranteed to find root'
            },
            'false_position': {
                'name': 'False Position Method',
                'convergence': 'Super-linear',
                'speed': 'Medium',
                'reliability': 'Very High',
                'description': 'Good balance of speed and reliability'
            },
            'newton_raphson': {
                'name': 'Newton-Raphson Method',
                'convergence': 'Quadratic',
                'speed': 'Fast',
                'reliability': 'High',
                'description': 'Fastest method when it works, requires derivative'
            }
        },
        'curve_fitting_methods': {
            'linear': {
                'name': 'Linear',
                'equation': 'y = mx + c',
                'degree': 1,
                'best_for': 'Steady trends'
            },
            'quadratic': {
                'name': 'Quadratic',
                'equation': 'y = ax² + bx + c',
                'degree': 2,
                'best_for': 'Curved trends'
            },
            'cubic': {
                'name': 'Cubic',
                'equation': 'y = ax³ + bx² + cx + d',
                'degree': 3,
                'best_for': 'Complex patterns'
            }
        },
        'error_analysis_types': {
            'truncation_error': 'Error from polynomial approximation',
            'roundoff_error': 'Error from computer arithmetic',
            'condition_number': 'Numerical stability measure'
        }
    }), 200
  
@app.route('/savings-goal')
def savings_goal():
    """Render the savings goal calculator page"""
    return render_template('savings-goal.html')

@app.route('/api/savings/calculate', methods=['POST'])
def calculate_savings_goal():
    """
    API endpoint to calculate required monthly savings using Bisection Method
    
    Expected JSON input:
    {
        "targetAmount": 100000,
        "years": 10,
        "annualRate": 6.0,
        "compounding": "monthly"
    }
    
    Returns:
    {
        "monthly_saving": 615.47,
        "total_contributions": 73856.40,
        "total_interest": 26143.60,
        "target_amount": 100000,
        "iterations": [...],
        "num_iterations": 12,
        "converged": true,
        "final_error": 0.01,
        "method": "Bisection Method"
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        target_amount = float(data.get('targetAmount', 0))
        years = float(data.get('years', 0))
        annual_rate = float(data.get('annualRate', 0))
        compounding = data.get('compounding', 'monthly')
        
        # Validate inputs
        if target_amount <= 0:
            return jsonify({'error': 'Target amount must be greater than 0'}), 400
        
        if years <= 0:
            return jsonify({'error': 'Time period must be greater than 0'}), 400
        
        if annual_rate < 0:
            return jsonify({'error': 'Interest rate cannot be negative'}), 400
        
        # Create calculator instance
        calculator = SavingsGoalCalculator(
            target_amount=target_amount,
            years=years,
            annual_rate=annual_rate,
            compounding=compounding
        )
        
        # Calculate required savings
        result = calculator.calculate()
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in calculate_savings_goal: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)