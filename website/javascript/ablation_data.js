window.renderAblationTable = function() {
    
    // GPT-3.5-turbo Ablation Data
    var gpt35Data = [
        { goal: "✗", process: "✗", alfred_sr: 44.4, alfred_gc: 47.5, habitat_sr: 43.6, habitat_gc: 50.4 },
        { goal: "✓", process: "✗", alfred_sr: 44.8, alfred_gc: 51.0, habitat_sr: 47.2, habitat_gc: 54.6 },
        { goal: "✗", process: "✓", alfred_sr: 42.4, alfred_gc: 47.5, habitat_sr: 45.2, habitat_gc: 51.3 },
        { goal: "✓", process: "✓", alfred_sr: 48.0, alfred_gc: 54.1, habitat_sr: 48.8, habitat_gc: 56.7, isWorldMind: true }
    ];

    // GPT-4.1-mini Ablation Data
    var gpt41Data = [
        { goal: "✗", process: "✗", alfred_sr: 41.2, alfred_gc: 47.5, habitat_sr: 41.6, habitat_gc: 47.4 },
        { goal: "✓", process: "✗", alfred_sr: 48.8, alfred_gc: 56.3, habitat_sr: 48.4, habitat_gc: 55.7 },
        { goal: "✗", process: "✓", alfred_sr: 46.4, alfred_gc: 51.5, habitat_sr: 47.6, habitat_gc: 54.0 },
        { goal: "✓", process: "✓", alfred_sr: 49.2, alfred_gc: 55.7, habitat_sr: 50.8, habitat_gc: 57.2, isWorldMind: true }
    ];

    var getColumnMax = function(data, field) {
        var max = -Infinity;
        for (var i = 0; i < data.length; i++) {
            if (data[i][field] > max) {
                max = data[i][field];
            }
        }
        return max;
    };


    var createValueFormatter = function(data, field) {
        var maxVal = getColumnMax(data, field);
        return function(cell) {
            var value = cell.getValue();
            var isMax = Math.abs(value - maxVal) < 0.01;
            var style = isMax ? "font-weight:800;color:#2d5a27;" : "";
            return "<span style='" + style + "'>" + parseFloat(value).toFixed(1) + "</span>";
        };
    };


    var checkFormatter = function(cell) {
        var value = cell.getValue();
        if (value === "✓") {
            return "<span style='color:#27ae60;font-weight:bold;'>✓</span>";
        } else {
            return "<span style='color:#999;'>✗</span>";
        }
    };


    var createAblationTable = function(containerId, data) {
        var container = document.getElementById(containerId);
        if (!container) return;

        var columns = [
            { title: "Goal", field: "goal", hozAlign: "center", minWidth: 80, headerSort: false, formatter: checkFormatter, headerHozAlign: "center" },
            { title: "Process", field: "process", hozAlign: "center", minWidth: 90, headerSort: false, formatter: checkFormatter, headerHozAlign: "center" },
            {
                title: "EB-ALFRED",
                headerHozAlign: "center",
                columns: [
                    { title: "SR", field: "alfred_sr", hozAlign: "center", minWidth: 70, headerSort: true, formatter: createValueFormatter(data, "alfred_sr"), headerHozAlign: "center" },
                    { title: "GC", field: "alfred_gc", hozAlign: "center", minWidth: 70, headerSort: true, formatter: createValueFormatter(data, "alfred_gc"), headerHozAlign: "center" }
                ]
            },
            {
                title: "EB-Habitat",
                headerHozAlign: "center",
                columns: [
                    { title: "SR", field: "habitat_sr", hozAlign: "center", minWidth: 70, headerSort: true, formatter: createValueFormatter(data, "habitat_sr"), headerHozAlign: "center" },
                    { title: "GC", field: "habitat_gc", hozAlign: "center", minWidth: 70, headerSort: true, formatter: createValueFormatter(data, "habitat_gc"), headerHozAlign: "center" }
                ]
            }
        ];

        new Tabulator("#" + containerId, {
            data: data,
            layout: "fitColumns",
            responsiveLayout: false,
            movableColumns: false,
            columnDefaults: { tooltip: true },
            columns: columns
        });
    };


    createAblationTable("ablation-table-gpt35", gpt35Data);
    createAblationTable("ablation-table-gpt41", gpt41Data);
};
