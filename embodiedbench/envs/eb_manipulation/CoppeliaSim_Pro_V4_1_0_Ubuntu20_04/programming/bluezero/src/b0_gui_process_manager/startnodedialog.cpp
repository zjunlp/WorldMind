#include "startnodedialog.h"

#include <QMessageBox>
#include <QDebug>

StartNodeDialog::StartNodeDialog(QWidget *parent) :
    QDialog(parent)
{
    setWindowFlags((windowFlags() | Qt::CustomizeWindowHint) & ~Qt::WindowCloseButtonHint);
    setupUi(this);
}

StartNodeDialog::~StartNodeDialog()
{
}

void StartNodeDialog::on_btnLaunch_clicked()
{
    btnLaunch->setEnabled(false);

    QString s(editArguments->toPlainText());
    QString cur;
    QStringList args;
    bool inQuotes = false;
    bool escaping = false;
    bool curEmpty = true;
    for(int i = 0; i < s.length(); i++)
    {
        QChar c(s[i]);
        if(escaping)
        {
            if(c == 'n') cur += '\n';
            else if(c == 'r') cur += '\r';
            else if(c == 't') cur += '\t';
            else cur += c;
            escaping = false;
        }
        else if(c == '"')
        {
            inQuotes = !inQuotes;
            curEmpty = false;
        }
        else if(c == '\\')
        {
            escaping = true;
        }
        else if(c == ' ' && !inQuotes)
        {
            args << cur;
            cur = "";
            curEmpty = true;
        }
        else
        {
            cur += c;
        }
    }
    if(!cur.isEmpty() || !curEmpty)
    {
        args << cur;
    }

    bool ok = false;
    int pid = -1;
    QString error;

    Q_EMIT startNode(comboHost->currentText(), comboProgram->currentText(), args);
}

void StartNodeDialog::on_btnCancel_clicked()
{
    reject();
}

void StartNodeDialog::displayStartNodeResult(bool ok, int pid, QString error)
{
    btnLaunch->setEnabled(true);

    if(ok)
    {
        accept();
    }
    else
    {
        QMessageBox msgBox;
        QString errMsg = QString("Failed to launch node:\n\n%1").arg(error);
        QMessageBox::critical(this, "Error", errMsg, QMessageBox::Ok, QMessageBox::Ok);
    }
}

