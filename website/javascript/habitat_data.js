window.renderHabitatTable = function() {
    
    // ==========================================
    // Success Rate (SR) Data
    // ==========================================
    var proprietaryDataSR = [
        { model: "GPT-4o",            avg: 54.0, base: 62.0, common: 50.0, complex: 62.0, visual: 48.0, spatial: 48.0 },
        { model: "GPT-4o-mini",       avg: 24.2, base: 30.0, common: 22.0, complex: 28.0, visual: 20.0, spatial: 21.0 },
        { model: "Claude-3.7-Sonnet", avg: 64.8, base: 70.0, common: 64.0, complex: 68.0, visual: 62.0, spatial: 60.0 },
        { model: "Gemini-1.5-Pro",    avg: 60.4, base: 68.0, common: 60.0, complex: 64.0, visual: 56.0, spatial: 54.0 },
        { model: "Llama-3.2-90B-Vis", avg: 32.4, base: 38.0, common: 30.0, complex: 36.0, visual: 28.0, spatial: 30.0 },
        { model: "InternVL2.5-78B",   avg: 34.6, base: 40.0, common: 34.0, complex: 38.0, visual: 18.0, spatial: 43.0 }
    ];
    var gpt35DataSR = [
        { model: "ReAct",         avg: 40.2, base: 48.0, common: 40.0, complex: 46.0, visual: 32.0, spatial: 35.0 },
        { model: "BoN",           avg: 38.6, base: 44.0, common: 38.0, complex: 42.0, visual: 36.0, spatial: 33.0 },
        { model: "SimuRA",        avg: 42.4, base: 48.0, common: 40.0, complex: 48.0, visual: 36.0, spatial: 40.0 },
        { model: "ReasoningBank", avg: 37.8, base: 46.0, common: 34.0, complex: 40.0, visual: 34.0, spatial: 35.0 },
        { model: "Synapse",       avg: 35.4, base: 36.0, common: 40.0, complex: 38.0, visual: 32.0, spatial: 31.0 },
        { model: "AWM",           avg: 36.8, base: 42.0, common: 30.0, complex: 42.0, visual: 38.0, spatial: 32.0 },
        { model: "WorldMind",     avg: 45.6, base: 54.0, common: 46.0, complex: 50.0, visual: 36.0, spatial: 42.0 }
    ];
    var gpt41DataSR = [
        { model: "ReAct",         avg: 38.4, base: 46.0, common: 36.0, complex: 42.0, visual: 36.0, spatial: 32.0 },
        { model: "BoN",           avg: 41.2, base: 44.0, common: 40.0, complex: 46.0, visual: 40.0, spatial: 36.0 },
        { model: "SimuRA",        avg: 43.8, base: 50.0, common: 42.0, complex: 50.0, visual: 38.0, spatial: 39.0 },
        { model: "ReasoningBank", avg: 35.6, base: 40.0, common: 32.0, complex: 38.0, visual: 32.0, spatial: 36.0 },
        { model: "Synapse",       avg: 34.2, base: 38.0, common: 30.0, complex: 40.0, visual: 32.0, spatial: 31.0 },
        { model: "AWM",           avg: 38.0, base: 40.0, common: 34.0, complex: 44.0, visual: 36.0, spatial: 36.0 },
        { model: "WorldMind",     avg: 47.6, base: 50.0, common: 54.0, complex: 50.0, visual: 40.0, spatial: 44.0 }
    ];

    // ==========================================
    // Goal Condition (GC) Data
    // ==========================================
    var proprietaryDataGC = [
        { model: "GPT-4o",            avg: 63.8, base: 72.0, common: 58.5, complex: 70.0, visual: 56.5, spatial: 62.0 },
        { model: "GPT-4o-mini",       avg: 32.5, base: 45.0, common: 33.0, complex: 40.5, visual: 30.0, spatial: 14.0 },
        { model: "Claude-3.7-Sonnet", avg: 64.5, base: 74.0, common: 65.0, complex: 74.0, visual: 60.0, spatial: 49.5 },
        { model: "Gemini-1.5-Pro",    avg: 66.2, base: 73.0, common: 64.5, complex: 73.5, visual: 61.0, spatial: 59.0 },
        { model: "Llama-3.2-90B-Vis", avg: 36.0, base: 42.0, common: 35.5, complex: 46.0, visual: 34.0, spatial: 22.5 },
        { model: "InternVL2.5-78B",   avg: 40.5, base: 41.5, common: 34.0, complex: 42.5, visual: 34.0, spatial: 50.5 }
    ];
    var gpt35DataGC = [
        { model: "ReAct",         avg: 48.2, base: 53.0, common: 50.5, complex: 52.0, visual: 40.0, spatial: 45.5 },
        { model: "BoN",           avg: 48.5, base: 52.0, common: 44.5, complex: 54.0, visual: 50.0, spatial: 42.0 },
        { model: "SimuRA",        avg: 51.8, base: 56.0, common: 46.0, complex: 58.0, visual: 46.5, spatial: 52.5 },
        { model: "ReasoningBank", avg: 45.0, base: 54.5, common: 40.0, complex: 45.0, visual: 42.0, spatial: 43.5 },
        { model: "Synapse",       avg: 42.0, base: 40.5, common: 48.5, complex: 41.5, visual: 40.0, spatial: 39.5 },
        { model: "AWM",           avg: 44.5, base: 50.0, common: 38.0, complex: 48.5, visual: 45.0, spatial: 41.0 },
        { model: "WorldMind",     avg: 53.5, base: 61.5, common: 51.0, complex: 60.0, visual: 42.0, spatial: 53.0 }
    ];
    var gpt41DataGC = [
        { model: "ReAct",         avg: 46.0, base: 52.5, common: 41.0, complex: 50.0, visual: 45.0, spatial: 41.5 },
        { model: "BoN",           avg: 48.0, base: 49.0, common: 46.5, complex: 52.5, visual: 46.0, spatial: 46.0 },
        { model: "SimuRA",        avg: 50.5, base: 58.0, common: 48.5, complex: 56.0, visual: 43.0, spatial: 47.0 },
        { model: "ReasoningBank", avg: 41.0, base: 44.5, common: 37.0, complex: 44.0, visual: 40.0, spatial: 39.5 },
        { model: "Synapse",       avg: 40.5, base: 40.0, common: 36.0, complex: 48.0, visual: 39.5, spatial: 39.0 },
        { model: "AWM",           avg: 44.2, base: 46.5, common: 40.0, complex: 50.0, visual: 42.5, spatial: 42.0 },
        { model: "WorldMind",     avg: 54.0, base: 59.0, common: 58.0, complex: 56.5, visual: 46.0, spatial: 50.5 }
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

    createTable("habitat-sr-proprietary", proprietaryDataSR);
    createTable("habitat-sr-gpt35", gpt35DataSR);
    createTable("habitat-sr-gpt41", gpt41DataSR);

    createTable("habitat-gc-proprietary", proprietaryDataGC);
    createTable("habitat-gc-gpt35", gpt35DataGC);
    createTable("habitat-gc-gpt41", gpt41DataGC);
};
