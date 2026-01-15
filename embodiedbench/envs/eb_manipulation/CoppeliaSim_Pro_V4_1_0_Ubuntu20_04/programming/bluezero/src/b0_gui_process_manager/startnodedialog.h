#ifndef STARTNODEDIALOG_H
#define STARTNODEDIALOG_H

#include <QDialog>
#include "ui_startnodedialog.h"

namespace Ui {
class StartNodeDialog;
}

class StartNodeDialog : public QDialog, public Ui::StartNodeDialog
{
    Q_OBJECT

public:
    explicit StartNodeDialog(QWidget *parent = nullptr);
    ~StartNodeDialog();

    inline void setPos(QPoint pos) { pos_ = pos; }

Q_SIGNALS:
    void startNode(QString host, QString program, QStringList args);

public Q_SLOTS:
    void displayStartNodeResult(bool ok, int pid, QString error);

private Q_SLOTS:
    void on_btnLaunch_clicked();
    void on_btnCancel_clicked();

private:
    QPoint pos_;

    friend class MainWindow;
};

#endif // STARTNODEDIALOG_H
