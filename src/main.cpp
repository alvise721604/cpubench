#include <QApplication>
#include <QStyleFactory>
#include <QDebug>

#include "picalculatorwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    qDebug() << QStyleFactory::keys();
    app.setStyle("windowsvista");
    PiCalculatorWindow window;
    window.show();

    return app.exec();
}
