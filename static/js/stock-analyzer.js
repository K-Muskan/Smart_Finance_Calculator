let mainChart, predictionChart, convergenceChart;
let analysisResults = null;

// Chart.js default configuration
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#e2e8f0';

// Tab switching for input methods
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tab}-tab`).classList.add('active');
    });
});

// Tab switching for results
document.querySelectorAll('.results-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        document.querySelectorAll('.results-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.results-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${tab}-section`).classList.add('active');
    });
});

// File upload
document.getElementById('fileUpload').addEventListener('click', () => {
    document.getElementById('csvFile').click();
});

document.getElementById('csvFile').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            document.getElementById('csvData').value = event.target.result;
            showSuccess('File uploaded successfully!');
        };
        reader.readAsText(file);
    }
});

// Load sample data
function loadSampleData() {
    const sampleData = `index,Date,Open,High,Low,Close,Adj Close,Volume
0,2010-01-04,7.6225,7.660714,7.585,7.643214,6.553025,493729600
1,2010-01-05,7.664286,7.699643,7.616071,7.656429,6.564355,601904800
2,2010-01-06,7.656429,7.686786,7.526786,7.534643,6.459939,552160000
3,2010-01-07,7.5625,7.571429,7.466071,7.520714,6.447999,477131200
4,2010-01-08,7.510714,7.571429,7.466429,7.570714,6.490865,447610800
5,2010-01-11,7.6,7.607143,7.444643,7.503929,6.433607,462229600
6,2010-01-12,7.471071,7.491786,7.372143,7.418571,6.360425,594459600
7,2010-01-13,7.423929,7.533214,7.289286,7.523214,6.450142,605892000
8,2010-01-14,7.503929,7.516429,7.465,7.479643,6.412785,432894000
9,2010-01-15,7.533214,7.557143,7.3525,7.354643,6.305614,594067600
10,2010-01-19,7.4175,7.491071,7.402857,7.478571,6.411866,474686400
11,2010-01-20,7.455357,7.535,7.419643,7.42,6.361734,569194400
12,2010-01-21,7.425357,7.446786,7.299643,7.305357,6.263443,591505200
13,2010-01-22,7.2775,7.305,7.058929,7.067143,6.059183,887413600
14,2010-01-25,7.075,7.217857,7.048571,7.162143,6.140629,679609200`;
    
    document.getElementById('csvData').value = sampleData;
    showSuccess('Sample data loaded successfully!');
}

// Form submission
document.getElementById('stockForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await analyzeStock();
});

async function analyzeStock() {
    const companyName = document.getElementById('companyName').value.trim();
    const method = document.getElementById('analysisMethod').value;
    const csvData = document.getElementById('csvData').value.trim();

    if (!companyName) {
        showError('Please enter a company name');
        return;
    }

    if (!csvData) {
        showError('Please upload a CSV file or paste CSV data');
        return;
    }

    // Clear previous root finding results
    document.getElementById('targetPrice').value = '';
    document.getElementById('rootFindingResults').style.display = 'none';
    document.getElementById('rootResultInfo').innerHTML = '';
    if (convergenceChart) {
        convergenceChart.destroy();
        convergenceChart = null;
    }
    // Also hide the price range hint
    const priceRangeHint = document.getElementById('priceRangeHint');
    if (priceRangeHint) {
        priceRangeHint.style.display = 'none';
    }

    const btn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('resultsContent');

    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="display: block; width: 20px; height: 20px;"></div><span>Analyzing...</span>';
    
    resultsSection.style.display = 'block';
    loading.classList.add('active');
    resultsContent.style.display = 'none';
    clearError();

    try {
        const response = await fetch('/api/stock/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                companyName: companyName,
                csvData: csvData,
                method: method
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        analysisResults = data;
        displayResults(data);
        showSuccess('Analysis completed successfully!');

        // Show additional sections
        document.getElementById('rootFindingSection').style.display = 'block';
        document.getElementById('errorAnalysisSection').style.display = 'block';

        // Automatically perform error analysis
        await performErrorAnalysis();

    } catch (error) {
        console.error('Error:', error);
        showError('Error: ' + error.message);
        resultsSection.style.display = 'none';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>🚀</span> Analyze Stock Trend';
        loading.classList.remove('active');
    }
}

function displayResults(data) {
    document.getElementById('resultsContent').style.display = 'block';
    document.getElementById('resultCompanyName').textContent = data.company_name;
    document.getElementById('methodBadge').textContent = data.curve_fitting.method;
    
    if (data.auto_selected) {
        document.getElementById('autoBadge').style.display = 'inline-block';
    } else {
        document.getElementById('autoBadge').style.display = 'none';
    }

    displayStatistics(data.statistics);
    displayCurveFittingResults(data.curve_fitting);
    displayCharts(data);
    displayDataTable(data);

    if (data.comparison) {
        displayComparison(data.comparison);
    }
}

function displayStatistics(stats) {
    const statsGrid = document.getElementById('statsGrid');
    
    const changeClass = stats.price_change >= 0 ? 'positive' : 'negative';
    const changeSign = stats.price_change >= 0 ? '+' : '';

    statsGrid.innerHTML = `
        <div class="summary-card">
            <h4>Mean Price</h4>
            <div class="value">$${stats.mean.toFixed(2)}</div>
        </div>
        <div class="summary-card">
            <h4>Price Range</h4>
            <div class="value" style="font-size: 1.2rem;">$${stats.min.toFixed(2)} - $${stats.max.toFixed(2)}</div>
        </div>
        <div class="summary-card">
            <h4>Std Deviation</h4>
            <div class="value">${stats.std_dev.toFixed(4)}</div>
        </div>
        <div class="summary-card">
            <h4>Total Change</h4>
            <div class="value">$${Math.abs(stats.price_change).toFixed(2)}</div>
            <div class="change ${changeClass}">${changeSign}${stats.price_change_percent.toFixed(2)}%</div>
        </div>
        <div class="summary-card">
            <h4>Data Points</h4>
            <div class="value">${stats.data_points}</div>
        </div>
    `;
}

function displayCurveFittingResults(curveFitting) {
    document.getElementById('accuracyValue').textContent = (curveFitting.r_squared * 100).toFixed(2) + '%';
    document.getElementById('equationText').textContent = curveFitting.equation;

    const trendIndicator = document.getElementById('trendIndicator');
    const trendIcons = {
        'Upward': '📈',
        'Downward': '📉',
        'Flat': '➡️'
    };

    trendIndicator.textContent = trendIcons[curveFitting.trend] + ' ' + curveFitting.trend + ' Trend';
    trendIndicator.className = 'trend-badge ' + curveFitting.trend.toLowerCase();
}

function displayCharts(data) {
    const dates = data.dates;
    const actualPrices = data.prices;
    const predictions = data.curve_fitting.predictions;
    const futurePredictions = data.curve_fitting.future_predictions;

    // Destroy existing charts
    if (mainChart) mainChart.destroy();
    if (predictionChart) predictionChart.destroy();

    // Main Chart
    const mainCtx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(mainCtx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Actual Price',
                    data: actualPrices,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.1
                },
                {
                    label: 'Fitted Curve',
                    data: predictions,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    borderDash: [8, 4],
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            family: "'Inter', sans-serif",
                            size: 12,
                            weight: '600'
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleFont: {
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 12
                    },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': $' + context.parsed.y.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: '#e2e8f0'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                },
                x: {
                    grid: {
                        color: '#e2e8f0'
                    },
                    ticks: {
                        font: {
                            size: 10
                        },
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });

    // Prediction Chart
    const futureLabels = Array.from({length: 30}, (_, i) => `Day +${i + 1}`);
    const predCtx = document.getElementById('predictionChart').getContext('2d');
    predictionChart = new Chart(predCtx, {
        type: 'line',
        data: {
            labels: futureLabels,
            datasets: [
                {
                    label: 'Predicted Price',
                    data: futurePredictions,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    fill: true
                }
            ]
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
                            family: "'Inter', sans-serif",
                            size: 12,
                            weight: '600'
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleFont: {
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 12
                    },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return 'Predicted: $' + context.parsed.y.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: '#e2e8f0'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                },
                x: {
                    grid: {
                        color: '#e2e8f0'
                    },
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

function displayDataTable(data) {
    const tbody = document.getElementById('dataTableBody');
    tbody.innerHTML = '';

    const dates = data.dates;
    const actualPrices = data.prices;
    const predictions = data.curve_fitting.predictions;
    const residuals = data.curve_fitting.residuals;

    for (let i = 0; i < dates.length; i++) {
        const accuracy = ((1 - Math.abs(residuals[i]) / actualPrices[i]) * 100).toFixed(2);
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${dates[i]}</td>
            <td>$${actualPrices[i].toFixed(4)}</td>
            <td>$${predictions[i].toFixed(4)}</td>
            <td>${residuals[i] >= 0 ? '+' : ''}${residuals[i].toFixed(4)}</td>
            <td>${accuracy}%</td>
        `;
        tbody.appendChild(row);
    }
}

function displayComparison(comparison) {
    const grid = document.getElementById('comparisonGrid');
    grid.innerHTML = '';

    comparison.models.forEach(model => {
        const isBest = model.name === comparison.best_model;
        const card = document.createElement('div');
        card.className = 'model-card' + (isBest ? ' best' : '');
        card.innerHTML = `
            <h4>
                ${model.name}
                ${isBest ? '<span class="best-indicator">BEST FIT</span>' : ''}
            </h4>
            <div class="r2-value">${(model.r_squared * 100).toFixed(2)}%</div>
            <div style="color: #64748b; font-size: 0.85rem;">R² Score</div>
        `;
        grid.appendChild(card);
    });
}

// Root Finding
async function findRoot() {
    const targetPrice = parseFloat(document.getElementById('targetPrice').value);
    
    if (!targetPrice || targetPrice <= 0) {
        showError('Please enter a valid target price');
        return;
    }

    if (!analysisResults) {
        showError('Please analyze data first');
        return;
    }

    try {
        const response = await fetch('/api/stock/root-finding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                companyName: analysisResults.company_name,
                dates: analysisResults.dates,
                prices: analysisResults.prices,
                method: analysisResults.method_used.toLowerCase(),
                targetPrice: targetPrice
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Root finding failed');
        }

        displayRootFindingResults(data);
        document.getElementById('rootFindingResults').style.display = 'block';
        showSuccess('Target price analysis completed!');
    } catch (error) {
        showError('Error in root finding: ' + error.message);
    }
}

function displayRootFindingResults(result) {
    const infoDiv = document.getElementById('rootResultInfo');
    
    if (result.success) {
        infoDiv.innerHTML = `
            <h4>✅ Target Found: ${result.method}</h4>
            <p style="margin-bottom: 15px;">${result.message}</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <strong style="display: block; color: #64748b; font-size: 0.85rem; margin-bottom: 5px;">DAYS FROM NOW</strong>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #10b981;">${result.days_from_last_data.toFixed(1)}</span>
                </div>
                <div>
                    <strong style="display: block; color: #64748b; font-size: 0.85rem; margin-bottom: 5px;">ITERATIONS</strong>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">${result.iterations}</span>
                </div>
                <div>
                    <strong style="display: block; color: #64748b; font-size: 0.85rem; margin-bottom: 5px;">FINAL ERROR</strong>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">${result.error.toFixed(6)}</span>
                </div>
                <div>
                    <strong style="display: block; color: #64748b; font-size: 0.85rem; margin-bottom: 5px;">CONVERGENCE</strong>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">${result.convergence_rate}</span>
                </div>
            </div>
        `;

        // Display convergence chart
        if (result.error_history && result.error_history.length > 0) {
            displayConvergenceChart(result.error_history);
        }
    } else {
        // Show helpful message with current price info if available
        let helpMessage = '';
        if (result.current_price) {
            helpMessage = `
                <div style="background: #eff6ff; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #2563eb;">
                    <p style="color: #1e40af; font-weight: 600; margin-bottom: 8px;">💡 Helpful Tip:</p>
                    <p style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                        Current latest price: <strong>${result.current_price.toFixed(2)}</strong><br>
                        Your target: <strong>${result.target_price.toFixed(2)}</strong><br>
                        For an upward trend, try a target price higher than ${result.current_price.toFixed(2)}
                    </p>
                </div>
            `;
        } else if (result.days_in_past) {
            helpMessage = `
                <div style="background: #fef3c7; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #f59e0b;">
                    <p style="color: #92400e; font-weight: 600; margin-bottom: 8px;">⚠️ Target is in the Past:</p>
                    <p style="color: #78350f; font-size: 0.9rem; line-height: 1.6;">
                        Based on the fitted curve, this price point would have occurred approximately <strong>${result.days_in_past.toFixed(1)} days ago</strong>. 
                        Try a higher target price for future predictions.
                    </p>
                </div>
            `;
        }
        
        infoDiv.innerHTML = `
            <h4>❌ Target Not Reachable</h4>
            <p style="color: #dc2626; margin-bottom: 10px;">${result.message}</p>
            <p style="color: #64748b;">Iterations completed: ${result.iterations}</p>
            ${helpMessage}
        `;
        
        // Still show convergence chart if available
        if (result.error_history && result.error_history.length > 0) {
            displayConvergenceChart(result.error_history);
        } else {
            document.querySelector('#rootFindingResults .chart-container').style.display = 'none';
        }
    }
}

function displayConvergenceChart(errorHistory) {
    if (convergenceChart) {
        convergenceChart.destroy();
    }

    const ctx = document.getElementById('convergenceChart').getContext('2d');
    const labels = Array.from({length: errorHistory.length}, (_, i) => i + 1);

    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Error',
                data: errorHistory,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 3,
                pointRadius: 4,
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    callbacks: {
                        label: function(context) {
                            return 'Error: ' + context.parsed.y.toFixed(6);
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Error (log scale)'
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Iteration'
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                }
            }
        }
    });

    document.querySelector('#rootFindingResults .chart-container').style.display = 'block';
}

// Error Analysis
async function performErrorAnalysis() {
    if (!analysisResults) return;

    try {
        const response = await fetch('/api/stock/truncation-error', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                companyName: analysisResults.company_name,
                dates: analysisResults.dates,
                prices: analysisResults.prices,
                method: analysisResults.method_used.toLowerCase()
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayErrorAnalysis(data);
        }
    } catch (error) {
        console.error('Error in error analysis:', error);
    }
}

function displayErrorAnalysis(results) {
    const statsGrid = document.getElementById('errorStatsGrid');
    const explanation = document.getElementById('errorExplanation');

    statsGrid.innerHTML = `
        <div class="summary-card">
            <h4>Max Error</h4>
            <div class="value">${results.max_truncation_error.toFixed(6)}</div>
        </div>
        <div class="summary-card">
            <h4>Mean Error</h4>
            <div class="value">${results.mean_truncation_error.toFixed(6)}</div>
        </div>
        <div class="summary-card">
            <h4>Error %</h4>
            <div class="value">${results.error_percentage.toFixed(4)}%</div>
        </div>
        <div class="summary-card">
            <h4>Current Model</h4>
            <div class="value" style="font-size: 1.2rem;">${results.current_model}</div>
        </div>
    `;

    explanation.innerHTML = `<p>${results.message}</p>`;
}

// Export Results
function exportResults() {
    if (!analysisResults) return;

    const data = analysisResults;
    let csv = 'Date,Actual Price,Fitted Price,Residual,Accuracy\n';

    for (let i = 0; i < data.dates.length; i++) {
        const accuracy = ((1 - Math.abs(data.curve_fitting.residuals[i]) / data.prices[i]) * 100).toFixed(2);
        csv += `${data.dates[i]},${data.prices[i]},${data.curve_fitting.predictions[i]},${data.curve_fitting.residuals[i]},${accuracy}%\n`;
    }

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.company_name.replace(/\s+/g, '_')}_analysis.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showSuccess('Results exported successfully!');
}

// Utility Functions
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.classList.add('active');
    setTimeout(() => {
        errorDiv.classList.remove('active');
    }, 5000);
}

function clearError() {
    document.getElementById('errorMessage').classList.remove('active');
}

function showSuccess(message) {
    console.log('Success:', message);
}