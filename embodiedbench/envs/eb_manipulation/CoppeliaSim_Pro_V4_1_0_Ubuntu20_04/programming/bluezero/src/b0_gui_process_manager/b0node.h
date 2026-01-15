#ifndef B0NODE_H__INCLUDED
#define B0NODE_H__INCLUDED

#include <QObject>
#include <QMap>
#include <QSet>
#include <b0/b0.h>
#include <b0/node.h>
#include <b0/subscriber.h>
#include <b0/service_client.h>
#include <b0/message/graph/graph.h>
#include <b0_process_manager/protocol.h>

class B0Node : public QObject
{
    Q_OBJECT

public:
    B0Node();

private:
    void onGraphChanged(const b0::message::graph::Graph &msg);
    void onActiveNodesChanged(const b0::process_manager::ActiveNodes &msg);

Q_SIGNALS:
    void finished();
    void graphChanged(b0::message::graph::Graph msg);
    void activeNodesChanged(b0::process_manager::ActiveNodes msg);
    void startNodeResult(bool ok, int pid, QString error);
    void stopNodeResult(bool ok, QString error);

public Q_SLOTS:
    void run();
    void startNode(QString host, QString program, QStringList args);
    void stopNode(QString host, int pid);

private:
    std::unique_ptr<b0::Node> node_;
    std::unique_ptr<b0::Subscriber> graph_sub_;
    std::unique_ptr<b0::Subscriber> active_nodes_sub_;
    std::unique_ptr<b0::ServiceClient> pm_cli_;
    QSet<QString> activeNodes_;
};

#endif // B0NODE_H__INCLUDED
