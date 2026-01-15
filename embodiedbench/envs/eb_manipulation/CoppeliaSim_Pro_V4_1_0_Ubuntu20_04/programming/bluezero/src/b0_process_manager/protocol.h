#ifndef B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED
#define B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED

#include <string>
#include <b0/message/message.h>

namespace b0
{

namespace process_manager
{

class StartProcessRequest : public b0::message::Message
{
public:
    std::string path;
    std::vector<std::string> args;

    std::string type() const override {return "b0::process_manager::StartProcessRequest";}
};

class StartProcessResponse : public b0::message::Message
{
public:
    bool success;
    boost::optional<std::string> error_message;
    boost::optional<int> pid;

    std::string type() const override {return "b0::process_manager::StartProcessResponse";}
};

class StopProcessRequest : public b0::message::Message
{
public:
    int pid;

    std::string type() const override {return "b0::process_manager::StopProcessRequest";}
};

class StopProcessResponse : public b0::message::Message
{
public:
    bool success;
    boost::optional<std::string> error_message;

    std::string type() const override {return "b0::process_manager::StopProcessResponse";}
};

class QueryProcessStatusRequest : public b0::message::Message
{
public:
    int pid;

    std::string type() const override {return "b0::process_manager::QueryProcessStatusRequest";}
};

class QueryProcessStatusResponse : public b0::message::Message
{
public:
    bool success;
    boost::optional<std::string> error_message;
    boost::optional<bool> running;
    boost::optional<int> exit_code;

    std::string type() const override {return "b0::process_manager::QueryProcessStatusResponse";}
};

class ListActiveProcessesRequest : public b0::message::Message
{
public:

    std::string type() const override {return "b0::process_manager::ListActiveProcessesRequest";}
};

class ListActiveProcessesResponse : public b0::message::Message
{
public:
    std::vector<int> pids;

    std::string type() const override {return "b0::process_manager::ListActiveProcessesResponse";}
};

class Request : public b0::message::Message
{
public:
    boost::optional<StartProcessRequest> start_process;
    boost::optional<StopProcessRequest> stop_process;
    boost::optional<QueryProcessStatusRequest> query_process_status;
    boost::optional<ListActiveProcessesRequest> list_active_processes;

    std::string type() const override {return "b0::process_manager::Request";}
};

class Response : public b0::message::Message
{
public:
    boost::optional<StartProcessResponse> start_process;
    boost::optional<StopProcessResponse> stop_process;
    boost::optional<QueryProcessStatusResponse> query_process_status;
    boost::optional<ListActiveProcessesResponse> list_active_processes;

    std::string type() const override {return "b0::process_manager::Response";}
};

class HUBRequest : public Request
{
public:
    std::string host_name;
};

class HUBResponse : public Response
{
public:
    bool success;
    boost::optional<std::string> error_message;

    inline void operator=(const Response &rhs)
    {
        Response::operator=(rhs);
    }
};

class Beacon : public b0::message::Message
{
public:
    std::string host_name;
    std::string node_name;
    std::string service_name;

    std::string type() const override {return "b0::process_manager::Beacon";}
};

class NodeActivity : public b0::message::Message
{
public:
    std::string host_name;
    std::string node_name;
    std::string service_name;
    int64_t last_active;

    std::string type() const override {return "b0::process_manager::NodeActivity";}
};

class ActiveNodes : public b0::message::Message
{
public:
    std::vector<NodeActivity> nodes;

    std::string type() const override {return "b0::process_manager::ActiveNodes";}
};

} // namespace process_manager

} // namespace b0

namespace spotify
{

namespace json
{

template <>
struct default_codec_t<b0::process_manager::StartProcessRequest> {
    static codec::object_t<b0::process_manager::StartProcessRequest> codec() {
        auto codec = codec::object<b0::process_manager::StartProcessRequest>();
        codec.required("path", &b0::process_manager::StartProcessRequest::path);
        codec.required("args", &b0::process_manager::StartProcessRequest::args);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::StopProcessRequest> {
    static codec::object_t<b0::process_manager::StopProcessRequest> codec() {
        auto codec = codec::object<b0::process_manager::StopProcessRequest>();
        codec.required("pid", &b0::process_manager::StopProcessRequest::pid);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::QueryProcessStatusRequest> {
    static codec::object_t<b0::process_manager::QueryProcessStatusRequest> codec() {
        auto codec = codec::object<b0::process_manager::QueryProcessStatusRequest>();
        codec.required("pid", &b0::process_manager::QueryProcessStatusRequest::pid);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::ListActiveProcessesRequest> {
    static codec::object_t<b0::process_manager::ListActiveProcessesRequest> codec() {
        auto codec = codec::object<b0::process_manager::ListActiveProcessesRequest>();
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::Request> {
    static codec::object_t<b0::process_manager::Request> codec() {
        auto codec = codec::object<b0::process_manager::Request>();
        codec.optional("start_process", &b0::process_manager::Request::start_process);
        codec.optional("stop_process", &b0::process_manager::Request::stop_process);
        codec.optional("query_process_status", &b0::process_manager::Request::query_process_status);
        codec.optional("list_active_processes", &b0::process_manager::Request::list_active_processes);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::HUBRequest> {
    static codec::object_t<b0::process_manager::HUBRequest> codec() {
        auto codec = codec::object<b0::process_manager::HUBRequest>();
        codec.required("host_name", &b0::process_manager::HUBRequest::host_name);
        codec.optional("start_process", &b0::process_manager::HUBRequest::start_process);
        codec.optional("stop_process", &b0::process_manager::HUBRequest::stop_process);
        codec.optional("query_process_status", &b0::process_manager::HUBRequest::query_process_status);
        codec.optional("list_active_processes", &b0::process_manager::HUBRequest::list_active_processes);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::StartProcessResponse> {
    static codec::object_t<b0::process_manager::StartProcessResponse> codec() {
        auto codec = codec::object<b0::process_manager::StartProcessResponse>();
        codec.required("success", &b0::process_manager::StartProcessResponse::success);
        codec.optional("error_message", &b0::process_manager::StartProcessResponse::error_message);
        codec.optional("pid", &b0::process_manager::StartProcessResponse::pid);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::StopProcessResponse> {
    static codec::object_t<b0::process_manager::StopProcessResponse> codec() {
        auto codec = codec::object<b0::process_manager::StopProcessResponse>();
        codec.required("success", &b0::process_manager::StopProcessResponse::success);
        codec.optional("error_message", &b0::process_manager::StopProcessResponse::error_message);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::QueryProcessStatusResponse> {
    static codec::object_t<b0::process_manager::QueryProcessStatusResponse> codec() {
        auto codec = codec::object<b0::process_manager::QueryProcessStatusResponse>();
        codec.required("success", &b0::process_manager::QueryProcessStatusResponse::success);
        codec.optional("error_message", &b0::process_manager::QueryProcessStatusResponse::error_message);
        codec.optional("running", &b0::process_manager::QueryProcessStatusResponse::running);
        codec.optional("exit_code", &b0::process_manager::QueryProcessStatusResponse::exit_code);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::ListActiveProcessesResponse> {
    static codec::object_t<b0::process_manager::ListActiveProcessesResponse> codec() {
        auto codec = codec::object<b0::process_manager::ListActiveProcessesResponse>();
        codec.required("pids", &b0::process_manager::ListActiveProcessesResponse::pids);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::Response> {
    static codec::object_t<b0::process_manager::Response> codec() {
        auto codec = codec::object<b0::process_manager::Response>();
        codec.optional("start_process", &b0::process_manager::Response::start_process);
        codec.optional("stop_process", &b0::process_manager::Response::stop_process);
        codec.optional("query_process_status", &b0::process_manager::Response::query_process_status);
        codec.optional("list_active_processes", &b0::process_manager::Response::list_active_processes);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::HUBResponse> {
    static codec::object_t<b0::process_manager::HUBResponse> codec() {
        auto codec = codec::object<b0::process_manager::HUBResponse>();
        codec.required("success", &b0::process_manager::HUBResponse::success);
        codec.optional("error_message", &b0::process_manager::HUBResponse::error_message);
        codec.optional("start_process", &b0::process_manager::HUBResponse::start_process);
        codec.optional("stop_process", &b0::process_manager::HUBResponse::stop_process);
        codec.optional("query_process_status", &b0::process_manager::HUBResponse::query_process_status);
        codec.optional("list_active_processes", &b0::process_manager::HUBResponse::list_active_processes);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::Beacon> {
    static codec::object_t<b0::process_manager::Beacon> codec() {
        auto codec = codec::object<b0::process_manager::Beacon>();
        codec.required("host_name", &b0::process_manager::Beacon::host_name);
        codec.required("node_name", &b0::process_manager::Beacon::node_name);
        codec.required("service_name", &b0::process_manager::Beacon::service_name);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::NodeActivity> {
    static codec::object_t<b0::process_manager::NodeActivity> codec() {
        auto codec = codec::object<b0::process_manager::NodeActivity>();
        codec.required("host_name", &b0::process_manager::NodeActivity::host_name);
        codec.required("node_name", &b0::process_manager::NodeActivity::node_name);
        codec.required("service_name", &b0::process_manager::NodeActivity::service_name);
        codec.required("last_active", &b0::process_manager::NodeActivity::last_active);
        return codec;
    }
};

template <>
struct default_codec_t<b0::process_manager::ActiveNodes> {
    static codec::object_t<b0::process_manager::ActiveNodes> codec() {
        auto codec = codec::object<b0::process_manager::ActiveNodes>();
        codec.required("nodes", &b0::process_manager::ActiveNodes::nodes);
        return codec;
    }
};

} // namespace json

} // namespace spotify

#endif // B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED
