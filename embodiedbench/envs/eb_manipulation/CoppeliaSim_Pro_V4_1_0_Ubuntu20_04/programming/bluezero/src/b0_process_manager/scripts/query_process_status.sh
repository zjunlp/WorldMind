B0_CONSOLE_LOGLEVEL=warn ./b0_service_call -s process_manager/control -c b0::process_manager::Request <<EOF \
    | grep -v ^Content-type: \
    | python -m json.tool
{
    "query_process_status": {
        "pid": $1
    }
}
EOF
