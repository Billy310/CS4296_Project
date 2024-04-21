
function createChart(ctx, label) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    text: "Percentage (%)"
                }
            }
        }
    });
}

var cpuChart = createChart(document.getElementById('cpuChart').getContext('2d'), 'CPU Usage');
var gpuChart = createChart(document.getElementById('gpuChart').getContext('2d'), 'GPU Usage');
var ramChart = createChart(document.getElementById('ramChart').getContext('2d'), 'RAM Usage');

function updateChart(chart, value) {
    chart.data.labels.push(new Date().toLocaleTimeString());
    chart.data.datasets[0].data.push(value);

    if (chart.data.labels.length > 10) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }

    chart.update();
}

function fetchData() {
    fetch('/monitor_data')
        .then(response => response.json())
        .then(data => {
            updateChart(cpuChart, data.cpu_usage);
            updateChart(gpuChart, data.gpu_usage);
            updateChart(ramChart, data.ram_usage);
        });
}

setInterval(fetchData, 3000);

