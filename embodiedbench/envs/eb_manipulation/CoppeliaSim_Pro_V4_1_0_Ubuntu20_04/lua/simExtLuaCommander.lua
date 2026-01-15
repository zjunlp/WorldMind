local simLuaCmd={}

html_help = [[
        <h2>LuaCommander plugin</h2>
        <p>Use this plugin for quick evaluation of Lua expressions.</p>

        <h3>Completion</h3>
        <p>Begin to type the name of a function (e.g. "sim.getObjectHa") and press TAB to automatically complete it. If there are multiple matches, repeatedly press TAB to cycle through completions, and Shift+TAB to cycle back.</p>

        <h3>Keyboard Shortcuts</h3>
        <p>
        <b>Ctrl+Alt+C</b>: focus the text input.<br/>
        <b>TAB</b>: complete symbol / cycle to next completion.<br/>
        <b>Shift+TAB</b>: cycle to previous completion.<br/>
        <b>Enter</b>: accept completion (also works with '(' and '.').<br/>
        <b>Esc</b>: clear text field.<br/>
        <b>Up/Down</b> arrows: navigate/search through command history.<br/>
        <b>Ctrl+L</b>: clear statusbar.
        </p>

        <h3>String Rendering Flags</h3>
        <p>There are some flags that control how the results are displayed. Those are input by adding a comment at the end of the line, containing as comma separated list of key=value pairs, e.g.: "luaExpression --flag1=10,flag2=test". Flags can be abbreviated by typing only the initial part, e.g. "pre" instead of "precision", down to any length, provided it is not ambiguous.</p>
        <ul>
            <li><b>depth</b>: (integer) limit the maximum depth when rendering a map-table.</li>
            <li><b>precision</b>: (integer) number of floating point digits.</li>
            <li><b>retvals</b>: (1, *) print all the returned values (*) or only the first (1).</li>
            <li><b>sort</b>: (k, t, tk, off) how to sort map-table entries:
                    k: sort by keys;
                    t: sort by type;
                    tk: sort by type, then by key;
                    off: don't sort at all.
            </li>
            <li><b>escape</b>: (0/1) enable/disable special character escaping.</li>
        </ul>
]]

function help()
    if ui then  
        simUI.destroy(ui)
        ui = nil
    else
        ui = simUI.create('<ui title="LuaCommander Plugin" closeable="true" on-close="help" modal="true" size="440,520"><text-browser text="'..string.gsub(html_help,'"','&quot;')..'" /></ui>')
    end
end

return simLuaCmd
