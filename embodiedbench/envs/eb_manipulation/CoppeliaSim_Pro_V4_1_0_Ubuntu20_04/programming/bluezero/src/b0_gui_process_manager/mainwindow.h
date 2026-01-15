#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include "b0node.h"

class MainWindow : public QMainWindow, public Ui::MainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(B0Node *node, QWidget *parent = nullptr);
    ~MainWindow();

    NodesView * nodesView() const;

private:
    B0Node *node_;
};

#endif // MAINWINDOW_H
