#ifndef NODESVIEW_H
#define NODESVIEW_H

#include <QObject>
#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QTextStream>
#include <QMenu>
#include <b0/message/graph/graph.h>
#include <b0_process_manager/protocol.h>
#include "startnodedialog.h"

class NodesView;

class Connection;

class AbstractItem : public QGraphicsItem
{
public:
    AbstractItem(NodesView *nodeView);
    virtual ~AbstractItem();

protected:
    NodesView *nodesView_;
    QColor outlineColor_;
};

class AbstractVertex : public AbstractItem
{
public:
    AbstractVertex(NodesView *nodeView, const QString &text, const QColor &color);
    virtual ~AbstractVertex();

    QPainterPath shape() const override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    QString text() const;

protected:
    QPointF pointOnBorderAtAngle(qreal angle) const;

private:
    QString text_;
    QColor color_;
    QPainterPath path_;
    QVector<Connection *> connections_;

    void computePath();

    friend class Connection;
};

class Node : public AbstractVertex
{
public:
    Node(NodesView *nodeView, const QString &text);

    inline QString host() const { return host_; }
    inline int pid() const { return pid_; }

    inline void setInfo(const QString &host, int pid) { host_ = host; pid_ = pid; }

private:
    QString host_;
    int pid_;
};

class AbstractSocket : public AbstractVertex
{
public:
    AbstractSocket(NodesView *nodeView, const QString &text, const QColor &color);
};

class Topic : public AbstractSocket
{
public:
    Topic(NodesView *nodeView, const QString &text);
};

class Service : public AbstractSocket
{
public:
    Service(NodesView *nodeView, const QString &text);
};

enum class Direction
{
    In,
    Out
};

class Connection : public AbstractItem
{
public:
    Connection(NodesView *nodeView, Node *node, AbstractSocket *sock, Direction dir);
    virtual ~Connection();

    QPainterPath shape() const override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    void update();
    AbstractVertex * source() const;
    AbstractVertex * destination() const;
    Node * node() const;
    AbstractSocket * socket() const;

private:
    Node *node_;
    AbstractSocket *socket_;
    Direction dir_;
    QLineF line_;
    QColor color_;
};

class NodesView : public QGraphicsView
{
    Q_OBJECT
public:
    NodesView(QWidget *parent = nullptr);

    Node * addNode(const QString &text);
    Topic * addTopic(const QString &text);
    Service * addService(const QString &text);
    Connection * addConnection(Node *node, AbstractSocket *socket, Direction dir);

    void raiseItem(QGraphicsItem *item);
    void arrangeItems();
    void toGraphviz(QTextStream &stream, QMap<QString, AbstractVertex *> &itemMap) const;

    void contextMenuEvent(QContextMenuEvent *event);

public Q_SLOTS:
    void setGraph(b0::message::graph::Graph msg);
    void setActiveNodes(b0::process_manager::ActiveNodes msg);

private Q_SLOTS:
    void onMenuStartNode();
    void onMenuStopNode();
    void onMenuArrangeItems();

Q_SIGNALS:
    void stopNode(const QString &host, int pid);

private:
    QGraphicsScene *scene_;
    QMenu *contextMenu_;
    QAction *actionStartNode_;
    QAction *actionStopNode_;
    QAction *actionInfo_[4];
    StartNodeDialog *startNodeDialog_;
    b0::message::graph::Graph graph_;

    friend class MainWindow;
};

#endif // NODESVIEW_H
