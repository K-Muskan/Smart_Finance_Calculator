let chart = null;
let currentResult = null;
let amortizationData = [];

function calculatePayment() {
    const principal = parseFloat(document.getElementById('principal').value);
    const annualRate = parseFloat(document.getElementById('interestRate').value);
    const loanTerm = parseFloat(document.getElementById('loanTerm').value);
    const termUnit = document.getElementById('termUnit').value;
    const compounding = document.getElementById('compounding').value;
    const method = document.getElementById('method').value;

    if (!principal || !annualRate || !loanTerm) {
        alert('Please fill in all required fields');
        return;
    }

    if (principal <= 0 || annualRate < 0 || loanTerm <= 0) {
        alert('Please enter valid positive values');
        return;
    }

    fetch('/api/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            principal: principal,
            annualRate: annualRate,
            loanTerm: loanTerm,
            termUnit: termUnit,
            compounding: compounding,
            method: method
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        currentResult = data;
        displayResults(data);
        displayIterations(data.iterations);
        fetchAmortization(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Calculation failed. Please try again.');
    });
}

function displayResults(data) {
    document.getElementById('monthlyPayment').textContent = 'Rs ' + data.monthlyPayment.toLocaleString('en-PK', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    document.getElementById('totalInterest').textContent = 'Rs ' + data.totalInterest.toLocaleString('en-PK', {minimumFractionDigits: 0, maximumFractionDigits: 0});
    document.getElementById('totalPaid').textContent = 'Rs ' + data.totalPaid.toLocaleString('en-PK', {minimumFractionDigits: 0, maximumFractionDigits: 0});
    document.getElementById('iterations').textContent = data.numIterations;
}

function displayIterations(iterations) {
    const logDiv = document.getElementById('iterationLog');
    const methodUsed = currentResult.method || 'Numerical Method';
    document.getElementById('iterationTitle').textContent = methodUsed + ' - Iteration Log';
    logDiv.innerHTML = '';

    iterations.forEach(iter => {
        const row = document.createElement('div');
        row.className = 'iteration-row';
        row.innerHTML = `
            <strong>Iteration ${iter.iteration + 1}:</strong> 
            Guess = Rs ${iter.guess.toFixed(2)} | 
            f(x) = ${iter.fx.toFixed(4)} | 
            Error = ${iter.error.toFixed(4)}%
        `;
        logDiv.appendChild(row);
    });
}

function fetchAmortization(data) {
    fetch('/api/amortization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            principal: data.principalAmount,
            monthlyPayment: data.monthlyPayment,
            monthlyRate: data.monthlyRate,
            totalMonths: data.totalMonths
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.schedule) {
            amortizationData = result.schedule;
            displayChart(result.schedule);
            populateAmortization(result.schedule);
        }
    })
    .catch(error => {
        console.error('Error fetching amortization:', error);
    });
}

function displayChart(schedule) {
    const ctx = document.getElementById('balanceChart').getContext('2d');
    
    if (chart) {
        chart.destroy();
    }

    const labels = schedule.map(item => item.month);
    const balances = schedule.map(item => item.balance);

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Remaining Balance (Rs)',
                data: balances,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true,
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            weight: 600
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return 'Rs ' + value.toLocaleString('en-PK');
                        }
                    }
                },
                x: {
                    ticks: {
                        maxTicksLimit: 12
                    }
                }
            }
        }
    });
}

function populateAmortization(schedule) {
    const tbody = document.getElementById('amortizationBody');
    tbody.innerHTML = '';

    schedule.forEach(item => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${item.month}</td>
            <td>Rs ${item.payment.toLocaleString('en-PK', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
            <td>Rs ${item.principal.toLocaleString('en-PK', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
            <td>Rs ${item.interest.toLocaleString('en-PK', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
            <td>Rs ${item.balance.toLocaleString('en-PK', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
        `;
    });
}

function toggleAmortization() {
    const card = document.getElementById('amortizationCard');
    if (card.style.display === 'none') {
        if (!currentResult) {
            alert('Please calculate payment first');
            return;
        }
        card.style.display = 'block';
        card.scrollIntoView({ behavior: 'smooth' });
    } else {
        card.style.display = 'none';
    }
}

function exportCSV() {
    if (amortizationData.length === 0) {
        alert('Please calculate payment first!');
        return;
    }

    fetch('/api/export-csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ schedule: amortizationData })
    })
    .then(response => response.json())
    .then(result => {
        if (result.csv) {
            const blob = new Blob([result.csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'amortization_schedule.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Export failed. Please try again.');
    });
}

function resetForm() {
    document.getElementById('loanForm').reset();
    document.getElementById('principal').value = '0';
    document.getElementById('interestRate').value = '0';
    document.getElementById('loanTerm').value = '0';
    document.getElementById('monthlyPayment').textContent = 'Rs 0.00';
    document.getElementById('totalInterest').textContent = 'Rs 0';
    document.getElementById('totalPaid').textContent = 'Rs 0';
    document.getElementById('iterations').textContent = '0';
    document.getElementById('iterationTitle').textContent = 'Iteration Log';
    document.getElementById('iterationLog').innerHTML = '<div class="iteration-row">Enter loan parameters and click "Calculate Payment" to see iteration details...</div>';
    document.getElementById('amortizationCard').style.display = 'none';
    
    if (chart) {
        chart.destroy();
        chart = null;
    }
    
    currentResult = null;
    amortizationData = [];
}