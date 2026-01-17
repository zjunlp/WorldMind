window.renderNavigationTable = function() {
    
    // Navigation Data
    var navigationData = [
        { subset: "Base",                gpt35_react: 56.7, gpt35_worldmind: 66.7, gpt41_react: 53.3, gpt41_worldmind: 56.7 },
        { subset: "Common Sense",        gpt35_react: 58.3, gpt35_worldmind: 55.0, gpt41_react: 61.7, gpt41_worldmind: 60.0 },
        { subset: "Complex Instruction", gpt35_react: 66.7, gpt35_worldmind: 68.3, gpt41_react: 55.0, gpt41_worldmind: 56.7 },
        { subset: "Visual Appearance",   gpt35_react: 43.3, gpt35_worldmind: 45.0, gpt41_react: 46.7, gpt41_worldmind: 48.3 },
        { subset: "Average",             gpt35_react: 56.3, gpt35_worldmind: 58.8, gpt41_react: 54.2, gpt41_worldmind: 55.4, isAverage: true }
    ];

    var container = document.getElementById("navigation-table");
    if (!container) return;

    var getRowMaxForGroup = function(row, fields) {
        var max = -Infinity;
        for (var i = 0; i < fields.length; i++) {
            if (row[fields[i]] > max) {
                max = row[fields[i]];
            }
        }
        return max;
    };

    var createFormatter = function(field, groupFields) {
        return function(cell) {
            var value = cell.getValue();
            var rowData = cell.getRow().getData();
            var rowMax = getRowMaxForGroup(rowData, groupFields);
            var isMax = Math.abs(value - rowMax) < 0.01;
            var style = "";
            
            if (rowData.isAverage) {
                style = "font-weight:600;";
                if (isMax) {
                    style = "font-weight:800;color:#2d5a27;";
                }
            } else if (isMax) {
                style = "font-weight:800;color:#2d5a27;";
            }
            
            return "<span style='" + style + "'>" + parseFloat(value).toFixed(1) + "</span>";
        };
    };

    var columns = [
        {
            title: "Task Subset",
            field: "subset",
            minWidth: 160,
            headerSort: false,
            hozAlign: "left",
            headerHozAlign: "center",
            formatter: function(cell) {
                var val = cell.getValue();
                var rowData = cell.getRow().getData();
                var style = rowData.isAverage ? "font-weight:700;" : "";
                return "<span style='" + style + "'>" + val + "</span>";
            }
        },
        {
            title: "GPT-3.5-turbo",
            headerHozAlign: "center",
            columns: [
                { title: "ReAct", field: "gpt35_react", hozAlign: "center", minWidth: 80, headerSort: true, headerHozAlign: "center", formatter: createFormatter("gpt35_react", ["gpt35_react", "gpt35_worldmind"]) },
                { title: "WorldMind", field: "gpt35_worldmind", hozAlign: "center", minWidth: 100, headerSort: true, headerHozAlign: "center", formatter: createFormatter("gpt35_worldmind", ["gpt35_react", "gpt35_worldmind"]) }
            ]
        },
        {
            title: "GPT-4.1-mini",
            headerHozAlign: "center",
            columns: [
                { title: "ReAct", field: "gpt41_react", hozAlign: "center", minWidth: 80, headerSort: true, headerHozAlign: "center", formatter: createFormatter("gpt41_react", ["gpt41_react", "gpt41_worldmind"]) },
                { title: "WorldMind", field: "gpt41_worldmind", hozAlign: "center", minWidth: 100, headerSort: true, headerHozAlign: "center", formatter: createFormatter("gpt41_worldmind", ["gpt41_react", "gpt41_worldmind"]) }
            ]
        }
    ];

    new Tabulator("#navigation-table", {
        data: navigationData,
        layout: "fitColumns",
        responsiveLayout: false,
        movableColumns: false,
        columnDefaults: { tooltip: true },
        columns: columns
    });
};
