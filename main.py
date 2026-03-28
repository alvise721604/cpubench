import sys
import time
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QGridLayout,
)

import calc
import calc_vec


class PiCalculatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Calcolo di PI-Greco con vari metodi"
        self.algo_choices = [
            "Leibniz",
            "Euler",
            "Riemann sinx/x Integral",
            "Gaussian Integral",
        ]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        layout = QGridLayout()
        self.setLayout(layout)

        self.advanced_check = QCheckBox("Usa Numpy/Vettorizzato")
        self.advanced_check.setEnabled(False)

        self.algo_choice = QComboBox()
        self.algo_choice.addItems(self.algo_choices)
        self.algo_choice.currentTextChanged.connect(self.on_algo_choice)

        self.label_iter = QLabel("Inserisci il numero di iterazioni:")
        self.entry_iter = QLineEdit()
        self.entry_iter.setText("10000")

        self.label_step = QLabel("Inserisci lo step:")
        self.entry_step = QLineEdit()
        self.entry_step.setText("0.001")

        self.calculate_button = QPushButton("Calcola")
        self.calculate_button.clicked.connect(self.on_calculate_button_click)

        self.result_label = QLabel("")
        self.timer_label = QLabel("")

        layout.addWidget(self.algo_choice, 0, 0, 1, 2)
        layout.addWidget(self.advanced_check, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.label_iter, 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_iter, 2, 1)

        layout.addWidget(self.label_step, 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_step, 3, 1)

        layout.addWidget(self.calculate_button, 4, 0, 1, 2)
        layout.addWidget(self.result_label, 5, 0, 1, 2)
        layout.addWidget(self.timer_label, 6, 0, 1, 2)

        layout.setColumnStretch(1, 1)
        layout.setRowStretch(4, 1)

        self.algo_choice.setCurrentIndex(0)
        self.on_algo_choice(self.algo_choice.currentText())

        self.resize_and_center()

    def resize_and_center(self):
        font_metrics = QFontMetrics(self.font())
        title_width = font_metrics.horizontalAdvance(self.title) + 5

        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        window_width = max(320, title_width)
        window_height = 250

        center_x = screen_geometry.x() + (screen_geometry.width() - window_width) // 2
        center_y = screen_geometry.y() + (screen_geometry.height() - window_height) // 2

        self.setGeometry(center_x, center_y, window_width, window_height)

    def show_warning(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    def on_calculate_button_click(self):
        try:
            iterations = int(self.entry_iter.text())
            if iterations < 1:
                self.result_label.setText("Errore: iterazioni deve essere almeno 1")
                return

            if iterations < 1000:
                self.show_warning(
                    "Attenzione",
                    "Attenzione: poche iterazioni (<1000) possono dare valori poco accurati di pi-greco",
                )

            step = float(self.entry_step.text())

            if step > 0.001:
                self.show_warning(
                    "Attenzione",
                    "Valori maggiori di 0.001 non sono raccomandati per un calcolo accurato di pi-greco.",
                )

            if step <= 0:
                self.result_label.setText("Errore: lo step non può essere zero o negativo")
                return

            algorithm = self.algo_choice.currentText()
            advanced_mode = self.advanced_check.isChecked()

            if algorithm == "Leibniz":
                print(f"-> Calcolo con metodo Leibniz: iterazioni={iterations}")
                result = 0.0
                start_time = time.time()
                for k in range(iterations):
                    result += (-1) ** k / (2 * k + 1)
                result *= 4
                duration = time.time() - start_time

            elif algorithm == "Euler":
                print(f"-> Calcolo con metodo Euler: iterazioni={iterations}")
                result = 0.0
                start_time = time.time()
                for k in range(iterations):
                    result += 1 / ((k + 1) * (k + 1))
                result *= 6
                result = np.sqrt(result)
                duration = time.time() - start_time

            elif algorithm == "Riemann sinx/x Integral":
                print(f"-> Calcolo con metodo Integrale xin(x)/x VECTORIZED: limite={int(self.entry_iter.text())}, step={float(self.entry_step.text())}")
                if advanced_mode:
                    duration, result = calc_vec.riemann_sinx_integral_vectorized(
                        limit=int(self.entry_iter.text()),
                        step=float(self.entry_step.text()),
                    )
                else:
                    print(f"-> Calcolo con metodo Integrale xin(x)/x NORMALE: limite={int(self.entry_iter.text())}, step={float(self.entry_step.text())}")
                    duration, result = calc.riemann_sinx_integral(
                        iterations=int(int(self.entry_iter.text()) / step),
                        step=float(self.entry_step.text()),
                    )

            elif algorithm == "Gaussian Integral":
                if not advanced_mode:
                    # Mantengo la logica del tuo sorgente così com'è
                    print(f"-> Calcolo con metodo Integrale Gaussiano NORMALE: limite={int(self.entry_iter.text())}, step={float(self.entry_step.text())}")
                    duration, result = calc.gaussian_integral(
                        iterations=int(int(self.entry_iter.text()) / step),
                        step=float(self.entry_step.text()),
                    )
                else:
                    print(f"-> Calcolo con metodo Integrale Gaussiano VETTORIZZATO: limite={int(self.entry_iter.text())}, step={float(self.entry_step.text())}")
                    duration, result = calc_vec.gaussian_integral_numpy_vectorized(
                        limit=int(self.entry_iter.text()),
                        step=float(self.entry_step.text()),
                    )
            else:
                self.result_label.setText("Errore: algoritmo non riconosciuto")
                return

            print(f"\tRisultato: {result:.5f}")
            print(f"\tTempo: {duration:.6f} secondi")

            self.result_label.setText(f"Risultato: {result:.5f}")
            self.timer_label.setText(f"Durata: {duration:.6f} secondi")

        except ValueError as e:
            self.result_label.setText(f"Errore: {str(e)}")

    def on_algo_choice(self, selected_algorithm: str):
        if selected_algorithm in ("Leibniz", "Euler"):
            self.entry_step.setEnabled(False)
            self.advanced_check.setEnabled(False)
            self.label_iter.setText("Inserisci il numero di iterazioni:")
            self.entry_iter.setText("10000")
        else:
            self.entry_step.setEnabled(True)
            self.advanced_check.setEnabled(True)
            self.label_iter.setText("Inserisci l'estremo d'integrazione")
            self.entry_iter.setText("10000")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PiCalculatorWindow()
    window.show()
    sys.exit(app.exec())