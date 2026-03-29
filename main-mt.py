import sys
import time
import threading
import queue
import numpy as np

from PyQt6.QtCore import Qt, QTimer
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
            # "GPU",
        ]

        self.worker_thread = None
        self.result_queue = queue.Queue()

        self.init_ui()

        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self.check_worker_result)
        self.queue_timer.start(100)

    # _______________________________________________________________________
    def reset_ui(self):
        self.entry_step.setEnabled(False)
        self.engine_choice.setEnabled(False)
        self.label_iter.setText("Inserisci il numero di iterazioni:")
        self.entry_iter.setText("50000000")
        self.entry_step.setText("0.001")
        self.algo_choice.setCurrentIndex(0)
        self.engine_choice.setCurrentIndex(0)

    # _______________________________________________________________________
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

        self.label_step = QLabel("Inserisci lo step d'integrazione:")
        self.entry_step = QLineEdit()
        self.entry_step.setText("0.001")

        self.calculate_button = QPushButton("Calcola")
        self.calculate_button.clicked.connect(self.on_calculate_button_click)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_button_click)

        self.result_label = QLabel("")
        self.timer_label = QLabel("")

        layout.addWidget(self.algo_choice, 0, 0, 1, 2)
        layout.addWidget(self.engine_choice, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.label_iter, 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_iter, 2, 1)

        layout.addWidget(self.label_step, 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.entry_step, 3, 1)

        layout.addWidget(self.calculate_button, 4, 0, 1, 1)
        layout.addWidget(self.reset_button, 4, 1, 1, 1)
        layout.addWidget(self.result_label, 5, 0, 1, 2)
        layout.addWidget(self.timer_label, 6, 0, 1, 2)

        layout.addWidget(
            self.label_copyright,
            99,
            0,
            1,
            2,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
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

        self.reset_ui()

        

    # _______________________________________________________________________
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

    # _______________________________________________________________________
    def show_warning(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    # _______________________________________________________________________
    def set_ui_busy(self, busy: bool):
        self.calculate_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.algo_choice.setEnabled(not busy)
        self.engine_choice.setEnabled(not busy and self.algo_choice.currentText() not in ("Leibniz", "Euler"))
        self.entry_iter.setEnabled(not busy)
        self.entry_step.setEnabled(not busy and self.algo_choice.currentText() not in ("Leibniz", "Euler"))

    # _______________________________________________________________________
    def worker_calculation(self, algorithm: str, engine: str, iterations: int, step: float):
        try:
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
                if engine == "CPU Numpy Vectorized":
                    print(f"-> Calcolo con metodo Integrale sin(x)/x VETTORIZZATO: limite={iterations}, step={step}")
                    duration, result = calc_vec.riemann_sinx_integral_vectorized(
                        limit=iterations,
                        step=step,
                    )
                else:
                    print(f"-> Calcolo con metodo Integrale sin(x)/x NORMALE: limite={iterations}, step={step}")
                    duration, result = calc.riemann_sinx_integral(
                        iterations=int(iterations / step),
                        step=step,
                    )

            elif algorithm == "Gaussian Integral":
                if engine == "CPU Normal":
                    print(f"-> Calcolo con metodo Integrale Gaussiano NORMALE: limite={iterations}, step={step}")
                    duration, result = calc.gaussian_integral(
                        iterations=int(iterations / step),
                        step=step,
                    )
                else:
                    print(f"-> Calcolo con metodo Integrale Gaussiano VETTORIZZATO: limite={iterations}, step={step}")
                    duration, result = calc_vec.gaussian_integral_numpy_vectorized(
                        limit=iterations,
                        step=step,
                    )
            else:
                raise ValueError(f"Algoritmo non supportato: {algorithm}")

            print(f"\tRisultato: {result:.5f}")
            print(f"\tTempo: {duration:.6f} secondi")

            self.result_queue.put({
                "ok": True,
                "result": result,
                "duration": duration,
            })

        except Exception as e:
            self.result_queue.put({
                "ok": False,
                "error": str(e),
            })

    # _______________________________________________________________________
    def on_calculate_button_click(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.show_warning("Attenzione", "Un calcolo è già in esecuzione.")
            return

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

            engine = self.engine_choice.currentText()
            algorithm = self.algo_choice.currentText()

            self.result_label.setText("Risultato: calcolo in corso...")
            self.timer_label.setText("Durata: ...")
            self.set_ui_busy(True)

            self.worker_thread = threading.Thread(
                target=self.worker_calculation,
                args=(algorithm, engine, iterations, step),
                daemon=True,
            )
            self.worker_thread.start()

        except ValueError as e:
            self.result_label.setText(f"Errore: {str(e)}")

    # _______________________________________________________________________
    def check_worker_result(self):
        try:
            message = self.result_queue.get_nowait()
        except queue.Empty:
            return

        self.set_ui_busy(False)

        if message["ok"]:
            result = message["result"]
            duration = message["duration"]
            self.result_label.setText(f"Risultato: {result:.5f}")
            self.timer_label.setText(f"Durata: {duration:.6f} secondi")
        else:
            self.result_label.setText(f"Errore: {message['error']}")
            self.timer_label.setText("Durata: ---")

    # _______________________________________________________________________
    def on_algo_choice(self, selected_algorithm: str):
        if selected_algorithm in ("Leibniz", "Euler"):
            self.entry_step.setEnabled(False)
            self.engine_choice.setEnabled(False)
            self.label_iter.setText("Inserisci il numero di iterazioni:")
        else:
            self.entry_step.setEnabled(True)
            self.engine_choice.setEnabled(True)
            self.label_iter.setText("Inserisci l'estremo d'integrazione")

    # _______________________________________________________________________
    def on_reset_button_click(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.show_warning("Attenzione", "Non puoi fare reset mentre il calcolo è in esecuzione.")
            return
        self.reset_ui()
        # self.entry_step.setEnabled(False)
        # self.engine_choice.setEnabled(False)
        # self.label_iter.setText("Inserisci il numero di iterazioni:")
        # self.entry_iter.setText("50000000")
        # self.entry_step.setText("0.001")
        # self.algo_choice.setCurrentIndex(0)
        # self.engine_choice.setCurrentIndex(0)
        # self.result_label.setText("Risultato: ---")
        # self.timer_label.setText("Durata: ---")


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = PiCalculatorWindow()
    window.show()
    sys.exit(app.exec())