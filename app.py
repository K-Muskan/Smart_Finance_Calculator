from flask import Flask, render_template, request, jsonify
from modules.loan_payment import LoanCalculator
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['JSON_SORT_KEYS'] = False

# ============================================
# ROUTES - Web Pages
# ============================================

@app.route('/')
def index():
    """Home page"""
    return render_template('loan-analyzer.html')

@app.route('/loan-analyzer')
def loan_analyzer():
    """Loan analyzer page with Newton-Raphson calculator"""
    return render_template('loan-analyzer.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# ============================================
# API ENDPOINTS - Calculations
# ============================================

@app.route('/api/calculate', methods=['POST'])
def calculate_payment():
    """
    API endpoint to calculate loan payment using Newton-Raphson method
    
    Expected JSON input:
    {
        "principal": 50000,
        "annualRate": 5.5,
        "loanTerm": 30,
        "termUnit": "years",
        "compounding": "monthly"
    }
    
    Returns:
    {
        "monthlyPayment": 283.87,
        "totalPaid": 102193.20,
        "totalInterest": 52193.20,
        "iterations": [...],
        "numIterations": 8,
        "converged": true,
        "monthlyRate": 0.00458,
        "totalMonths": 360
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['principal', 'annualRate', 'loanTerm']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract and validate parameters
        principal = float(data.get('principal', 0))
        annual_rate = float(data.get('annualRate', 0))
        loan_term = float(data.get('loanTerm', 0))
        term_unit = data.get('termUnit', 'years')
        compounding = data.get('compounding', 'monthly')
        
        # Validate inputs
        if principal <= 0:
            return jsonify({'error': 'Principal amount must be greater than 0'}), 400
        if annual_rate < 0:
            return jsonify({'error': 'Interest rate cannot be negative'}), 400
        if loan_term <= 0:
            return jsonify({'error': 'Loan term must be greater than 0'}), 400
        
        # Create calculator instance
        calculator = LoanCalculator(
            principal=principal,
            annual_rate=annual_rate,
            loan_term=loan_term,
            term_unit=term_unit,
            compounding=compounding
        )
        
        # Calculate payment using Newton-Raphson method
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
        
        # Extract parameters
        principal = float(data.get('principal', 0))
        monthly_payment = float(data.get('monthlyPayment', 0))
        monthly_rate = float(data.get('monthlyRate', 0))
        total_months = int(data.get('totalMonths', 0))
        
        # Validate inputs
        if principal <= 0 or monthly_payment <= 0 or total_months <= 0:
            return jsonify({'error': 'Invalid parameters for amortization'}), 400
        
        # Create calculator and generate schedule
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
        import io
        import csv
        
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


@app.route('/api/compare-methods', methods=['POST'])
def compare_methods():
    """
    API endpoint to compare different calculation scenarios
    (For future expansion with multiple methods)
    """
    try:
        data = request.get_json()
        
        principal = float(data.get('principal', 0))
        annual_rate = float(data.get('annualRate', 0))
        loan_term = float(data.get('loanTerm', 0))
        term_unit = data.get('termUnit', 'years')
        compounding = data.get('compounding', 'monthly')
        
        calculator = LoanCalculator(
            principal=principal,
            annual_rate=annual_rate,
            loan_term=loan_term,
            term_unit=term_unit,
            compounding=compounding
        )
        
        # Calculate using Newton-Raphson
        result = calculator.calculate_newton_raphson()
        
        return jsonify({
            'newtonRaphson': result,
            'method': 'Newton-Raphson Method'
        }), 200
    
    except Exception as e:
        print(f"Error in compare_methods: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500



# MAIN APPLICATION
if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )