var colorFormatterSubgoal = function (cell, formatterParams) {
    var value = cell.getValue();
    if (value === "-") { return value; }

    var defaults = { min: 0.0, max: 100.0, startColor: { r: 255, g: 255, b: 255 }, endColor: { r: 245, g: 232, b: 221 } };
    var min = (formatterParams && formatterParams.min !== undefined) ? formatterParams.min : defaults.min;
    var max = (formatterParams && formatterParams.max !== undefined) ? formatterParams.max : defaults.max;

    var normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
    var r = Math.floor(255 + (245 - 255) * normalizedValue);
    var g = Math.floor(255 + (232 - 255) * normalizedValue);
    var b = Math.floor(255 + (221 - 255) * normalizedValue);

    return "<span style='display:block;width:100%;height:100%;background-color:rgb(" + r + "," + g + "," + b + ");'>" + parseFloat(value).toFixed(1) + "</span>";
};

var colorFormatterActionSeq = function (cell, formatterParams) {
    var value = cell.getValue();
    if (value === "-") { return value; }

    var defaults = { min: 0.0, max: 100.0 };
    var min = (formatterParams && formatterParams.min !== undefined) ? formatterParams.min : defaults.min;
    var max = (formatterParams && formatterParams.max !== undefined) ? formatterParams.max : defaults.max;

    var normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
    var r = Math.floor(255 + (204 - 255) * normalizedValue);
    var g = Math.floor(255 + (211 - 255) * normalizedValue);
    var b = Math.floor(255 + (202 - 255) * normalizedValue);

    return "<span style='display:block;width:100%;height:100%;background-color:rgb(" + r + "," + g + "," + b + ");'>" + parseFloat(value).toFixed(1) + "</span>";
};

var colorFormatterTrans = function (cell, formatterParams) {
    var value = cell.getValue();
    if (value === "-") { return value; }

    var defaults = { min: 0.0, max: 100.0 };
    var min = (formatterParams && formatterParams.min !== undefined) ? formatterParams.min : defaults.min;
    var max = (formatterParams && formatterParams.max !== undefined) ? formatterParams.max : defaults.max;

    var normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
    var r = Math.floor(255 + (181 - 255) * normalizedValue);
    var g = Math.floor(255 + (192 - 255) * normalizedValue);
    var b = Math.floor(255 + (208 - 255) * normalizedValue);

    return "<span style='display:block;width:100%;height:100%;background-color:rgb(" + r + "," + g + "," + b + ");'>" + parseFloat(value).toFixed(1) + "</span>";
};

var colorFormatterGoalInt = function (cell, formatterParams) {
    var value = cell.getValue();
    if (value === "-") { return value; }

    var defaults = { min: 0.0, max: 100.0 };
    var min = (formatterParams && formatterParams.min !== undefined) ? formatterParams.min : defaults.min;
    var max = (formatterParams && formatterParams.max !== undefined) ? formatterParams.max : defaults.max;

    var normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
    var r = Math.floor(255 + (238 - 255) * normalizedValue);
    var g = Math.floor(255 + (211 - 255) * normalizedValue);
    var b = Math.floor(255 + (217 - 255) * normalizedValue);

    return "<span style='display:block;width:100%;height:100%;background-color:rgb(" + r + "," + g + "," + b + ");'>" + parseFloat(value).toFixed(1) + "</span>";
};

var barColorFn = function (value) {
    var normalizedValue = Math.max(0, Math.min(1, value / 80));
    var r = Math.floor(255 + (206 - 255) * normalizedValue);
    var g = Math.floor(255 + (212 - 255) * normalizedValue);
    var b = Math.floor(255 + (218 - 255) * normalizedValue);
    return 'rgba(' + r + ',' + g + ',' + b + ',0.9)';
};

var getColumnMinMax = function(data, field) {
    var values = [];
    for (var i = 0; i < data.length; i++) {
        if (data[i][field] !== "-" && data[i][field] !== undefined) {
            values.push(Number(data[i][field]));
        }
    }
    if (values.length === 0) return { min: 0, max: 100 };
    return { min: Math.min.apply(null, values), max: Math.max.apply(null, values) };
};


window.renderAlfredTable = function() {
    
    // ==========================================
    // Success Rate (SR) Data
    // ==========================================
    var proprietaryDataSR = [
        { model: "GPT-4o",            avg: 56.8, base: 64.0, common: 54.0, complex: 68.0, visual: 46.0, spatial: 52.0 },
        { model: "GPT-4o-mini",       avg: 28.8, base: 34.0, common: 28.0, complex: 36.0, visual: 24.0, spatial: 22.0 },
        { model: "Claude-3.7-Sonnet", avg: 67.2, base: 68.0, common: 68.0, complex: 70.0, visual: 68.0, spatial: 62.0 },
        { model: "Gemini-1.5-Pro",    avg: 63.2, base: 70.0, common: 64.0, complex: 72.0, visual: 58.0, spatial: 52.0 },
        { model: "Llama-3.2-90B-Vis", avg: 35.2, base: 38.0, common: 34.0, complex: 44.0, visual: 28.0, spatial: 32.0 },
        { model: "InternVL2.5-78B",   avg: 37.0, base: 41.0, common: 40.0, complex: 39.0, visual: 16.0, spatial: 49.0 }
    ];
    var gpt35DataSR = [
        { model: "ReAct",         avg: 44.4, base: 52.0, common: 48.0, complex: 52.0, visual: 32.0, spatial: 38.0 },
        { model: "BoN",           avg: 42.8, base: 46.0, common: 42.0, complex: 50.0, visual: 42.0, spatial: 34.0 },
        { model: "SimuRA",        avg: 45.2, base: 50.0, common: 42.0, complex: 54.0, visual: 38.0, spatial: 42.0 },
        { model: "ReasoningBank", avg: 41.6, base: 50.0, common: 36.0, complex: 44.0, visual: 36.0, spatial: 42.0 },
        { model: "Synapse",       avg: 38.8, base: 38.0, common: 46.0, complex: 40.0, visual: 36.0, spatial: 34.0 },
        { model: "AWM",           avg: 40.0, base: 46.0, common: 32.0, complex: 48.0, visual: 40.0, spatial: 34.0 },
        { model: "WorldMind",     avg: 48.0, base: 58.0, common: 48.0, complex: 56.0, visual: 34.0, spatial: 44.0 }
    ];
    var gpt41DataSR = [
        { model: "ReAct",         avg: 41.2, base: 50.0, common: 40.0, complex: 46.0, visual: 38.0, spatial: 32.0 },
        { model: "BoN",           avg: 44.4, base: 46.0, common: 44.0, complex: 50.0, visual: 42.0, spatial: 40.0 },
        { model: "SimuRA",        avg: 45.6, base: 52.0, common: 44.0, complex: 54.0, visual: 38.0, spatial: 40.0 },
        { model: "ReasoningBank", avg: 38.0, base: 42.0, common: 36.0, complex: 42.0, visual: 34.0, spatial: 36.0 },
        { model: "Synapse",       avg: 37.2, base: 40.0, common: 32.0, complex: 44.0, visual: 36.0, spatial: 34.0 },
        { model: "AWM",           avg: 41.2, base: 44.0, common: 36.0, complex: 48.0, visual: 38.0, spatial: 40.0 },
        { model: "WorldMind",     avg: 49.2, base: 50.0, common: 58.0, complex: 54.0, visual: 42.0, spatial: 42.0 }
    ];

    // ==========================================
    // Goal Condition (GC) Data
    // ==========================================
    var proprietaryDataGC = [
        { model: "GPT-4o",            avg: 65.1, base: 74.0, common: 60.3, complex: 74.0, visual: 58.3, spatial: 61.3 },
        { model: "GPT-4o-mini",       avg: 34.3, base: 47.8, common: 35.3, complex: 43.5, visual: 33.3, spatial: 29.0 },
        { model: "Claude-3.7-Sonnet", avg: 65.3, base: 72.0, common: 66.0, complex: 76.7, visual: 63.0, spatial: 59.7 },
        { model: "Gemini-1.5-Pro",    avg: 67.4, base: 74.3, common: 66.7, complex: 76.5, visual: 62.8, spatial: 59.0 },
        { model: "Llama-3.2-90B-Vis", avg: 37.6, base: 43.7, common: 37.3, complex: 49.2, visual: 35.3, spatial: 36.0 },
        { model: "InternVL2.5-78B",   avg: 41.0, base: 42.3, common: 35.3, complex: 43.3, visual: 35.7, spatial: 40.3 }
    ];
    var gpt35DataGC = [
        { model: "ReAct",             avg: 50.4, base: 55.3, common: 53.5, complex: 55.3, visual: 42.7, spatial: 45.0 },
        { model: "BoN",               avg: 50.4, base: 54.2, common: 46.5, complex: 56.5, visual: 52.0, spatial: 42.8 },
        { model: "SimuRA",            avg: 53.6, base: 57.8, common: 47.8, complex: 59.7, visual: 48.5, spatial: 54.3 },
        { model: "ReasoningBank",     avg: 47.6, base: 57.5, common: 41.5, complex: 47.0, visual: 44.2, spatial: 48.0 },
        { model: "Synapse",           avg: 43.6, base: 42.5, common: 51.3, complex: 42.7, visual: 42.0, spatial: 39.7 },
        { model: "AWM",               avg: 46.2, base: 53.2, common: 39.2, complex: 50.7, visual: 47.0, spatial: 41.0 },
        { model: "WorldMind",         avg: 54.1, base: 63.0, common: 52.7, complex: 61.0, visual: 41.7, spatial: 52.0 }
    ];
    var gpt41DataGC = [
        { model: "ReAct",             avg: 47.5, base: 55.3, common: 42.8, complex: 52.2, visual: 47.2, spatial: 39.8 },
        { model: "BoN",               avg: 49.5, base: 50.8, common: 48.3, complex: 54.7, visual: 48.7, spatial: 45.0 },
        { model: "SimuRA",            avg: 52.2, base: 61.0, common: 50.3, complex: 58.2, visual: 45.3, spatial: 46.3 },
        { model: "ReasoningBank",     avg: 42.6, base: 46.7, common: 38.8, complex: 45.8, visual: 41.5, spatial: 40.3 },
        { model: "Synapse",           avg: 42.2, base: 41.2, common: 37.5, complex: 49.5, visual: 41.3, spatial: 41.7 },
        { model: "AWM",               avg: 46.0, base: 48.3, common: 42.0, complex: 52.5, visual: 44.3, spatial: 42.7 },
        { model: "WorldMind",         avg: 55.7, base: 61.0, common: 61.0, complex: 58.8, visual: 48.0, spatial: 49.7 }
    ];

    var createTable = function(containerId, data) {
        var container = document.getElementById(containerId);
        if (!container) return;

        var baseMinMax = getColumnMinMax(data, "base");
        var commonMinMax = getColumnMinMax(data, "common");
        var complexMinMax = getColumnMinMax(data, "complex");
        var visualMinMax = getColumnMinMax(data, "visual");
        var spatialMinMax = getColumnMinMax(data, "spatial");

        var columns = [
            {
                title: "Model",
                field: "model",
                widthGrow: 1.5,
                minWidth: 140,
                headerSort: true,
                formatter: function(cell) {
                    var val = cell.getValue();
                    if (val === "WorldMind") {
                        return "<span style='font-weight:800;color:#2d5a27;'>" + val + "</span>";
                    }
                    return val;
                }
            },
            {
                title: "Avg.",
                field: "avg",
                minWidth: 85,
                formatter: "progress",
                formatterParams: { min: 0, max: 80, legend: true, color: barColorFn },
                headerSort: true,
                sorter: "number"
            },
            { title: "Base", field: "base", hozAlign: "center", minWidth: 65, headerSort: true, formatter: colorFormatterSubgoal, formatterParams: { min: baseMinMax.min, max: baseMinMax.max } },
            { title: "Common", field: "common", hozAlign: "center", minWidth: 75, headerSort: true, formatter: colorFormatterActionSeq, formatterParams: { min: commonMinMax.min, max: commonMinMax.max } },
            { title: "Complex", field: "complex", hozAlign: "center", minWidth: 75, headerSort: true, formatter: colorFormatterTrans, formatterParams: { min: complexMinMax.min, max: complexMinMax.max } },
            { title: "Visual", field: "visual", hozAlign: "center", minWidth: 65, headerSort: true, formatter: colorFormatterGoalInt, formatterParams: { min: visualMinMax.min, max: visualMinMax.max } },
            { title: "Spatial", field: "spatial", hozAlign: "center", minWidth: 65, headerSort: true, formatter: colorFormatterSubgoal, formatterParams: { min: spatialMinMax.min, max: spatialMinMax.max } }
        ];

        new Tabulator("#" + containerId, {
            data: data,
            layout: "fitColumns",
            responsiveLayout: false,
            movableColumns: false,
            initialSort: [{ column: "avg", dir: "desc" }],
            columnDefaults: { tooltip: true },
            columns: columns
        });
    };

    createTable("alfred-sr-proprietary", proprietaryDataSR);
    createTable("alfred-sr-gpt35", gpt35DataSR);
    createTable("alfred-sr-gpt41", gpt41DataSR);

    createTable("alfred-gc-proprietary", proprietaryDataGC);
    createTable("alfred-gc-gpt35", gpt35DataGC);
    createTable("alfred-gc-gpt41", gpt41DataGC);
};
