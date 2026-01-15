#include "b0node.h"

#include <QDebug>

B0Node::B0Node()
{
}

void B0Node::onGraphChanged(const b0::message::graph::Graph &msg)
{
    Q_EMIT graphChanged(msg);
}

void B0Node::onActiveNodesChanged(const b0::process_manager::ActiveNodes &msg)
{
    Q_EMIT activeNodesChanged(msg);
}

void B0Node::run()
{
    b0::init();

    node_.reset(new b0::Node("b0_gui_process_manager"));
    graph_sub_.reset(new b0::Subscriber(node_.get(), "graph", static_cast<b0::Subscriber::CallbackMsg<b0::message::graph::Graph> >(boost::bind(&B0Node::onGraphChanged, this, _1))));
    active_nodes_sub_.reset(new b0::Subscriber(node_.get(), "process_manager_hub/active_nodes", static_cast<b0::Subscriber::CallbackMsg<b0::process_manager::ActiveNodes> >(boost::bind(&B0Node::onActiveNodesChanged, this, _1))));
    pm_cli_.reset(new b0::ServiceClient(node_.get(), "process_manager_hub/control"));

    node_->init();
    node_->spin();
    node_->cleanup();

    Q_EMIT finished();
}

void B0Node::startNode(QString host, QString program, QStringList args)
{
    b0::process_manager::HUBRequest req;
    req.host_name = host.toStdString();
    req.start_process.emplace();
    req.start_process->path = program.toStdString();
    for(auto arg : args)
        req.start_process->args.push_back(arg.toStdString());

    b0::process_manager::HUBResponse resp;

    pm_cli_->call(req, resp);

    // error can happen at HUB layer...
    if(!resp.success)
    {
        QString errMsg;
        if(resp.error_message)
            errMsg = QString::fromStdString(*resp.error_message);
        Q_EMIT startNodeResult(false, -1, errMsg);
        return;
    }

    // ...or at target Process Manager layer:
    if(!resp.start_process->success)
    {
        QString errMsg;
        if(resp.start_process->error_message)
            errMsg = QString::fromStdString(*resp.start_process->error_message);
        Q_EMIT startNodeResult(false, -1, errMsg);
        return;
    }

    // or not happen at all:
    Q_EMIT startNodeResult(true, *resp.start_process->pid, "");
}

void B0Node::stopNode(QString host, int pid)
{
    b0::process_manager::HUBRequest req;
    req.host_name = host.toStdString();
    req.stop_process.emplace();
    req.stop_process->pid = pid;

    b0::process_manager::HUBResponse resp;

    pm_cli_->call(req, resp);

    // error can happen at HUB layer...
    if(!resp.success)
    {
        QString errMsg;
        if(resp.error_message)
            errMsg = QString::fromStdString(*resp.error_message);
        Q_EMIT stopNodeResult(false, errMsg);
        return;
    }

    // ...or at target Process Manager layer:
    if(!resp.stop_process->success)
    {
        QString errMsg;
        if(resp.stop_process->error_message)
            errMsg = QString::fromStdString(*resp.stop_process->error_message);
        Q_EMIT stopNodeResult(false, errMsg);
        return;
    }

    // or not happen at all:
    Q_EMIT stopNodeResult(true, "");
}
