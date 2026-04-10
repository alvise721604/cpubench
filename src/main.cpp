#include <QApplication>
#include <QStyleFactory>
#include <QDebug>

#include "window.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    qDebug() << QStyleFactory::keys();
    app.setStyle("windowsvista");
    Window window;
    window.show();

    return app.exec();
}
