#include <QApplication>

#include "picalculatorwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    PiCalculatorWindow window;
    window.show();

    return app.exec();
}
