#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])

// include this file after calling the SIM_PLUGIN(...) macro

#py for cmd in plugin.commands:
void `cmd.name`(SScriptCallBack *p, const char *cmd, `cmd.name`_in *in, `cmd.name`_out *out)
{
    simPlugin->`cmd.name`(in, out);
}
#py endfor

