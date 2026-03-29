import sys
import time
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QGridLayout,
)

import calc
import calc_vec
import calc_mps


class CalculationWorker(QThread):
    finished = pyqtSignal(float, float)   # duration, result
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, algorithm: str, engine: str, iterations: int, step: float):
        super().__init__()
        self.algorithm = algorithm
        self.engine = engine
        self.iterations = iterations
        self.step = step
        self._stop_requested = False
        self.worker = None

    def request_stop(self):
        self._stop_requested = True

    def _check_stop(self):
        if self._stop_requested:
            raise InterruptedError("Calcolo interrotto dall'utente")

    def run(self):
        try:
            algorithm = self.algorithm
            engine = self.engine
            iterations = self.iterations
            step = self.step

            if algorithm == "Leibniz":
                print(f"-> Calcolo con metodo Leibniz: iterazioni={iterations}")
                result = 0.0
                start_time = time.time()
                for k in range(iterations):
                    if k % 1000 == 0:
                        self._check_stop()
                    result += (-1) ** k / (2 * k + 1)
                result *= 4
                duration = time.time() - start_time

            elif algorithm == "Euler":
                print(f"-> Calcolo con metodo Euler: iterazioni={iterations}")
                result = 0.0
                start_time = time.time()
                for k in range(iterations):
                    if k % 1000 == 0:
                        self._check_stop()
                    result += 1 / ((k + 1) * (k + 1))
                result *= 6
                result = np.sqrt(result)
                duration = time.time() - start_time

            elif algorithm == "Riemann sinx/x Integral":
                self._check_stop()
                if engine == "CPU Numpy Vectorized":
                    print(
                        f"-> Calcolo con metodo Integrale sin(x)/x VETTORIZZATO: "
                        f"limite={iterations}, step={step}"
                    )
                    duration, result = calc_vec.riemann_sinx_integral_vectorized(
                        limit=iterations,
                        step=step,
                    )
                else:
                    print(
                        f"-> Calcolo con metodo Integrale sin(x)/x NORMALE: "
                        f"limite={iterations}, step={step}"
                    )
                    duration, result = calc.riemann_sinx_integral(
                        iterations=int(iterations / step),
                        step=step,
                    )
                self._check_stop()

            elif algorithm == "Gaussian Integral":
                self._check_stop()
                if engine == "CPU Normal":
                    print(
                        f"-> Calcolo con metodo Integrale Gaussiano NORMALE: "
                        f"limite={iterations}, step={step}"
                    )
                    duration, result = calc.gaussian_integral(
                        iterations=int(iterations / step),
                        step=step,
                    )
                else:
                    print(
                        f"-> Calcolo con metodo Integrale Gaussiano VETTORIZZATO: "
                        f"limite={iterations}, step={step}"
                    )
                    duration, result = calc_vec.gaussian_integral_numpy_vectorized(
                        limit=iterations,
                        step=step,
                    )
                self._check_stop()

            else:
                raise ValueError("Algoritmo non riconosciuto")

            print(f"\tRisultato: {result:.5f}")
            print(f"\tTempo: {duration:.6f} secondi")
            self.finished.emit(duration, result)

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            self.error.emit(str(e))


class PiCalculatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Calcolo di π con vari metodi"
        self.algo_choices = [
            "Leibniz",
            "Euler",
            "Riemann sinx/x Integral",
            "Gaussian Integral",
        ]
        self.engine_choices = [
            "CPU Normal",
            "CPU Numpy Vectorized",
            "GPU",
        ]

        self.worker = None
        self.init_ui()

    #_______________________________________________________________________
    def init_ui(self):
        self.setWindowTitle(self.title)

        layout = QGridLayout()
        self.setLayout(layout)

        self.label_copyright = QLabel("(C) Alvise Dorigo)")

        self.algo_choice = QComboBox()
        self.algo_choice.addItems(self.algo_choices)
        self.algo_choice.currentTextChanged.connect(self.on_algo_choice)

        self.engine_choice = QComboBox()
        self.engine_choice.addItems(self.engine_choices)

        self.label_iter = QLabel("Inserisci il numero di iterazioni:")
        self.entry_iter = QLineEdit()
        self.entry_iter.setText("10000")

        self.label_step = QLabel("Inserisci lo step:")
        self.entry_step = QLineEdit()
        self.entry_step.setText("0.001")

        self.calculate_button = QPushButton("Calcola")
        self.calculate_button.clicked.connect(self.on_calculate_button_click)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_button_click)

        self.result_label = QLabel("")
        self.timer_label = QLabel("")

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)      
        self.stop_button.clicked.connect(self.on_stop_button_click)

        layout.addWidget(self.algo_choice, 0, 0, 1, 2)
        layout.addWidget(self.engine_choice, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.label_iter, 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_iter, 2, 1)

        layout.addWidget(self.label_step, 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_step, 3, 1)

        layout.addWidget(self.calculate_button, 4, 0, 1, 1)
        layout.addWidget(self.reset_button, 4, 1, 1, 1)
        layout.addWidget(self.stop_button, 5, 0, 1, 2)
        layout.addWidget(self.result_label, 6, 0, 1, 2)
        layout.addWidget(self.timer_label, 7, 0, 1, 2)

        layout.addWidget(
            self.label_copyright,
            99, 0, 1, 2,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        )
        layout.setRowStretch(98, 1)

        layout.setColumnStretch(1, 1)
        layout.setRowStretch(4, 1)

        self.algo_choice.setCurrentIndex(0)
        self.on_algo_choice(self.algo_choice.currentText())

        self.resize_and_center()
        self.setMinimumSize(350, 300)
        self.result_label.setText("Risultato: ---")
        self.timer_label.setText("Durata: ---")

    #_______________________________________________________________________
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

    #_______________________________________________________________________
    def show_warning(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    #_______________________________________________________________________
    def on_calculate_button_click(self):
        self.result_label.setText("Risultato: calcolo in corso...")
        self.timer_label.setText("Durata: calcolo in corso...")
        self.calculate_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        try:
            iterations = int(self.entry_iter.text())
            if iterations < 1:
                self.result_label.setText("Errore: iterazioni deve essere almeno 1")
                self.timer_label.setText("Durata: ---")
                self.calculate_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                self.stop_button.setEnabled(False)
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
                self.timer_label.setText("Durata: ---")
                self.calculate_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return

            engine = self.engine_choice.currentText()
            algorithm = self.algo_choice.currentText()

            self.worker = CalculationWorker(
                algorithm=algorithm,
                engine=engine,
                iterations=iterations,
                step=step,
            )
            self.worker.finished.connect(self.on_calculation_finished)
            self.worker.error.connect(self.on_calculation_error)
            self.worker.stopped.connect(self.on_calculation_stopped)
            self.worker.start()

        except ValueError as e:
            self.result_label.setText(f"Errore: {str(e)}")
            self.timer_label.setText("Durata: ---")
            self.calculate_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        except ValueError as e:
            self.result_label.setText(f"Errore: {str(e)}")
            self.timer_label.setText("Durata: ---")
            self.calculate_button.setEnabled(True)
            self.reset_button.setEnabled(True)

    #_______________________________________________________________________
    def on_calculation_finished(self, duration: float, result: float):
        self.result_label.setText(f"Risultato: {result:.5f}")
        self.timer_label.setText(f"Durata: {duration:.6f} secondi")
        self.calculate_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    #_______________________________________________________________________
    def on_calculation_error(self, message: str):
        self.result_label.setText(f"Errore: {message}")
        self.timer_label.setText("Durata: ---")
        self.calculate_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    #_______________________________________________________________________
    def on_stop_button_click(self):
        if self.worker is not None:
            self.result_label.setText("Risultato: interruzione richiesta...")
            self.timer_label.setText("Durata: arresto in corso...")
            self.stop_button.setEnabled(False)
            self.worker.request_stop()

    #_______________________________________________________________________
    def on_calculation_stopped(self):
        self.result_label.setText("Risultato: calcolo interrotto")
        self.timer_label.setText("Durata: ---")
        self.calculate_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    #_______________________________________________________________________
    def on_algo_choice(self, selected_algorithm: str):
        if selected_algorithm in ("Leibniz", "Euler"):
            self.entry_step.setEnabled(False)
            self.engine_choice.setEnabled(False)
            self.label_iter.setText("Inserisci il numero di iterazioni:")
        else:
            self.entry_step.setEnabled(True)
            self.engine_choice.setEnabled(True)
            self.label_iter.setText("Inserisci l'estremo d'integrazione")

    #_______________________________________________________________________
    def on_reset_button_click(self):
        self.result_label.setText("Risultato: ---")
        self.timer_label.setText("Durata: ---")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = PiCalculatorWindow()
    window.show()
    sys.exit(app.exec())
