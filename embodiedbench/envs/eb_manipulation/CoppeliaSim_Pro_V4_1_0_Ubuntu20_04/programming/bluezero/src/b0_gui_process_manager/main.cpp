#include "mainwindow.h"
#include "b0node.h"
#include <QApplication>
#include <QThread>
#include <QDebug>
#include <b0/message/graph/graph.h>
#include <b0_process_manager/protocol.h>

class B0NodeWorker : public QObject
{
    Q_OBJECT

public:
    B0NodeWorker(QThread *thread)
    {
        B0Node *node = &node_;
        QObject::connect(thread, &QThread::started, [=]() {node->run();});
        QObject::connect(node, &B0Node::finished, thread, &QThread::quit);
        QObject::connect(node, &B0Node::finished, node, &B0Node::deleteLater);
        QObject::connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    }

    B0Node node_;
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    qRegisterMetaType<b0::message::graph::Graph>("b0::message::graph::Graph");
    qRegisterMetaType<b0::process_manager::ActiveNodes>("b0::process_manager::ActiveNodes");

    QThread *thread = new QThread();
    B0NodeWorker *worker = new B0NodeWorker(thread);
    worker->moveToThread(thread);
    thread->start();

    MainWindow mainWindow(&worker->node_);
    mainWindow.show();

    return a.exec();
}
