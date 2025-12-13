from flask import Flask, render_template, request, jsonify
from modules.loan_payment import LoanCalculator
import traceback
import io
import csv

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)