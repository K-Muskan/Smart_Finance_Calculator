// Loan Payment Calculator - Flask Integration
let amortizationData = [];
let balanceChart = null;

// Calculate Payment using Flask Backend
async function calculatePayment() {
    try {
        // Get form values
        const principal = parseFloat(document.getElementById('principal').value);
        const annualRate = parseFloat(document.getElementById('interestRate').value);
        const loanTerm = parseFloat(document.getElementById('loanTerm').value);
        const termUnit = document.getElementById('termUnit').value;
        const compounding = document.getElementById('compounding').value;
        const method = document.getElementById('method').value;

        // Validate inputs
        if (isNaN(principal) || isNaN(annualRate) || isNaN(loanTerm)) {
            alert('Please enter valid numbers for all fields');
            return;
        }

        // Prepare data for API
        const data = {
            principal: principal,
            annualRate: annualRate,
            loanTerm: loanTerm,
            termUnit: termUnit,
            compounding: compounding,
            method: method
        };

        // Call Flask API
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Calculation failed');
        }

        const result = await response.json();

        // Display results
        document.getElementById('monthlyPayment').textContent = 
            '$' + result.monthlyPayment.toFixed(2);
        document.getElementById('totalInterest').textContent = 
            '$' + result.totalInterest.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        document.getElementById('totalPaid').textContent = 
            '$' + result.totalPaid.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        document.getElementById('iterations').textContent = result.numIterations;

        // Display iteration log
        displayIterationLog(result.iterations, result.converged);

        // Get amortization schedule
        await getAmortizationSchedule(
            principal,
            result.monthlyPayment,
            result.monthlyRate,
            result.totalMonths
        );

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during calculation. Please try again.');
    }
}

// Display iteration log
function displayIterationLog(iterations, converged) {
    const log = document.getElementById('iterationLog');
    let html = '';
    
    iterations.forEach(iter => {
        html += `<div class="iteration-row">Iteration ${iter.iteration}: guess = $${iter.guess.toFixed(2)}, f(x) = ${iter.fx.toFixed(2)}, error = ${iter.error.toFixed(4)}%</div>`;
    });
    
    if (converged) {
        html += `<div class="iteration-row" style="color: #28a745; font-weight: bold;">✓ Converged successfully!</div>`;
    } else {
        html += `<div class="iteration-row" style="color: orange;">⚠ Max iterations reached</div>`;
    }
    
    log.innerHTML = html;
}

// Get amortization schedule from Flask
async function getAmortizationSchedule(principal, monthlyPayment, monthlyRate, totalMonths) {
    try {
        const response = await fetch('/api/amortization', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                principal: principal,
                monthlyPayment: monthlyPayment,
                monthlyRate: monthlyRate,
                totalMonths: totalMonths
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get amortization schedule');
        }

        const result = await response.json();
        amortizationData = result.schedule;

        // Update chart
        updateBalanceChart(amortizationData);

    } catch (error) {
        console.error('Error:', error);
    }
}

// Update balance chart
function updateBalanceChart(schedule) {
    const ctx = document.getElementById('balanceChart');
    
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Destroy existing chart
    if (balanceChart) {
        balanceChart.destroy();
    }
    
    // Sample data (show every 12th month for clarity if > 60 months)
    const sampleRate = schedule.length > 60 ? 12 : 1;
    const sampledData = schedule.filter((_, i) => i % sampleRate === 0 || i === schedule.length - 1);
    
    balanceChart = new Chart(context, {
        type: 'line',
        data: {
            labels: sampledData.map(d => d.month),
            datasets: [{
                label: 'Remaining Balance',
                data: sampledData.map(d => d.balance),
                borderColor: '#0f8b8d',
                backgroundColor: 'rgba(15, 139, 141, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Balance: $' + context.parsed.y.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Month'
                    }
                }
            }
        }
    });
}

// Reset form
function resetForm() {
    document.getElementById('loanForm').reset();
    document.getElementById('principal').value = '50000';
    document.getElementById('interestRate').value = '5.5';
    document.getElementById('loanTerm').value = '30';
    document.getElementById('monthlyPayment').textContent = '$0.00';
    document.getElementById('totalInterest').textContent = '$0';
    document.getElementById('totalPaid').textContent = '$0';
    document.getElementById('iterations').textContent = '0';
    document.getElementById('iterationLog').innerHTML = 
        '<div class="iteration-row">Click "Calculate Payment" to see iteration details...</div>';
    
    if (balanceChart) {
        balanceChart.destroy();
        balanceChart = null;
    }
    
    document.getElementById('amortizationCard').style.display = 'none';
    amortizationData = [];
}

// Toggle amortization schedule
function toggleAmortization() {
    const card = document.getElementById('amortizationCard');
    
    if (amortizationData.length === 0) {
        alert('Please calculate payment first!');
        return;
    }
    
    card.style.display = card.style.display === 'none' ? 'block' : 'none';
    
    if (card.style.display === 'block') {
        displayAmortizationTable();
    }
}

// Display amortization table
function displayAmortizationTable() {
    const tbody = document.getElementById('amortizationBody');
    let html = '';
    
    // Show first 12 months, then sample every 12th month
    const displayData = amortizationData.filter((d, i) => 
        i < 12 || i % 12 === 0 || i === amortizationData.length - 1
    );
    
    displayData.forEach(row => {
        html += `
            <tr>
                <td>${row.month}</td>
                <td>$${row.payment.toFixed(2)}</td>
                <td>$${row.principal.toFixed(2)}</td>
                <td>$${row.interest.toFixed(2)}</td>
                <td>$${row.balance.toFixed(2)}</td>
            </tr>
        `;
    });
    
    if (amortizationData.length > displayData.length) {
        html += `<tr><td colspan="5" style="text-align: center; color: var(--text-light);">
            ... ${amortizationData.length - displayData.length} more rows ...
        </td></tr>`;
    }
    
    tbody.innerHTML = html;
}

// Export to CSV
async function exportCSV() {
    if (amortizationData.length === 0) {
        alert('Please calculate payment first!');
        return;
    }

    try {
        const response = await fetch('/api/export-csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                schedule: amortizationData
            })
        });

        if (!response.ok) {
            throw new Error('Export failed');
        }

        const result = await response.json();
        
        // Create download link
        const blob = new Blob([result.csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'amortization_schedule.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error:', error);
        alert('Export failed. Please try again.');
    }
}

// Hamburger menu toggle
const hamburger = document.getElementById('hamburger');
if (hamburger) {
    hamburger.addEventListener('click', () => {
        const navLinks = document.querySelector('.nav-links');
        navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
    });
}