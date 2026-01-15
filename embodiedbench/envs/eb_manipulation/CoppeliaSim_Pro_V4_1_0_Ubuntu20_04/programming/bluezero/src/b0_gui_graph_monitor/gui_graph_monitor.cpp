#include <b0/node.h>
#include <b0/subscriber.h>
#include <b0/message/graph/graph.h>
#include <b0/utils/graphviz.h>

#include <QApplication>
#include <QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QTimer>

class GraphConsoleWindow : public QMainWindow
{
public:
    GraphConsoleWindow(b0::Node &node)
        : QMainWindow(),
          node_(node)
    {
        setWindowTitle("BlueZero graph console");

        imageWidget = new QLabel;
        imageWidget->setBackgroundRole(QPalette::Base);
        imageWidget->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        imageWidget->setScaledContents(true);

        scrollArea = new QScrollArea;
        scrollArea->setBackgroundRole(QPalette::Dark);
        scrollArea->setWidget(imageWidget);

        setCentralWidget(scrollArea);

        QTimer *timer = new QTimer(this);
        connect(timer, &QTimer::timeout, [this](){this->node_.spinOnce();});
        timer->start(100);
    }

    void onGraphChanged(const b0::message::graph::Graph &graph)
    {
        b0::graph::toGraphviz(graph, "graph.gv");

        if(b0::graph::renderGraphviz("graph.gv", "graph.png") == 0)
        {
            QPixmap img("graph.png");
            imageWidget->setPixmap(img);
            imageWidget->adjustSize();
        }
    }

private:
    b0::Node &node_;
    QScrollArea *scrollArea;
    QLabel *imageWidget;
};

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    QApplication app(argc, argv);

    b0::Node graphConsoleNode("gui_graph_monitor");

    GraphConsoleWindow graphConsoleWindow(graphConsoleNode);

    b0::Subscriber logSub(&graphConsoleNode, "graph", &GraphConsoleWindow::onGraphChanged, &graphConsoleWindow);

    graphConsoleNode.init();

    graphConsoleWindow.resize(800, 340);
    graphConsoleWindow.show();

    int ret = app.exec();

    graphConsoleNode.cleanup();

    return ret;
}

