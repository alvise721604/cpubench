import os
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
    QPushButton,
    QComboBox,
    QMessageBox,
    QGridLayout,
)

import calc

LEIBNIZ_ITERATIONS = 800_000_000
EULER_ITERATIONS = 400_000_000

RIEMANN_GAUSS_LIMIT = 1000
RIEMANN_STEP = 1e-06

FBELLARD_ITERATIONS = 10_000_000

CORES = os.cpu_count()

class PiCalculatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Calcolo di π con vari metodi"
        self.algo_choices = [
            "Leibniz",
            "Euler",
            "F. Bellard",
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
        self.engine_choice.setEnabled(False)
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

        self.calculate_button = QPushButton("Calcola")
        self.calculate_button.clicked.connect(self.on_calculate_button_click)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_button_click)

        self.result_label = QLabel("")
        self.timer_label = QLabel("")

        layout.addWidget(self.algo_choice, 0, 0, 1, 2)
        layout.addWidget(self.engine_choice, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.calculate_button, 2, 0, 1, 1)
        layout.addWidget(self.reset_button, 2, 1, 1, 1)
        layout.addWidget(self.result_label, 3, 0, 1, 2)
        layout.addWidget(self.timer_label, 4, 0, 1, 2)

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

    # _______________________________________________________________________
    def worker_calculation( self, algorithm: str, engine: str ):
        try:

            if algorithm not in self.algo_choices:
                raise ValueError(f"Algoritmo non supportato: {algorithm}")

            if algorithm == "Leibniz":
                print(f"-> Calcolo con metodo Leibniz mp: iterazioni={EULER_ITERATIONS}")
                start_time = time.time()
                result = calc.pi_leibniz_multiprocessing(iterations=EULER_ITERATIONS, num_procs=CORES)
                duration = time.time() - start_time

            if algorithm == "Euler":
                print(f"-> Calcolo con metodo Euler: iterazioni={EULER_ITERATIONS}")
                start_time = time.time()
                result = calc.pi_euler_multiprocessing(iterations=EULER_ITERATIONS, num_procs=CORES)
                duration = time.time() - start_time

            if algorithm == "F. Bellard":
                print(f"-> Calcolo con metodo Fabrice Bellard: iterazioni={FBELLARD_ITERATIONS}")
                start_time = time.time()
                result = calc.pi_fabrice_bellard(iterations=FBELLARD_ITERATIONS)#, num_procs=CORES)
                duration = time.time() - start_time

            if algorithm == "Gaussian Integral":
                if engine == "CPU Normal":
                    print(f"-> Calcolo con metodo Integrale Gaussiano NORMALE: limite={RIEMANN_GAUSS_LIMIT}, step={RIEMANN_STEP}")
                    start_time = time.time()
                    result = calc.gaussian_integral(
                        iterations=int(RIEMANN_GAUSS_LIMIT / RIEMANN_STEP),
                        step=RIEMANN_STEP,
                    )
                    duration = time.time() - start_time

                else:
                    print(f"-> Calcolo con metodo Integrale Gaussiano VETTORIZZATO: limite={RIEMANN_GAUSS_LIMIT}, step={RIEMANN_STEP}")
                    start_time = time.time()
                    result = calc.gaussian_integral_numpy_vectorized(
                        limit=RIEMANN_GAUSS_LIMIT,
                        step=RIEMANN_STEP,
                    )
                    duration = time.time() - start_time

            print(f"\tRisultato: {result:.8f}")
            if duration < 1:
                print(f"\tTempo: {duration:.6f} secondi")
            else:
                print(f"\tTempo: {duration:.3f} secondi")

            self.result_queue.put({
                "ok": True,
                "result": result,
                "duration": duration,
            })

        except Exception as e:
            print(f"Error={e}")
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
            engine = self.engine_choice.currentText()
            algorithm = self.algo_choice.currentText()

            self.result_label.setText("Risultato: calcolo in corso...")
            self.timer_label.setText("Durata: ...")
            self.set_ui_busy(True)

            self.worker_thread = threading.Thread(
                target=self.worker_calculation,
                args=(algorithm, engine),
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
        if selected_algorithm in ("Leibniz", "Euler", "F. Bellard"):
            self.engine_choice.setEnabled(False)
        else:
            self.engine_choice.setEnabled(True)

    # _______________________________________________________________________
    def on_reset_button_click(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.show_warning("Attenzione", "Non puoi fare reset mentre il calcolo è in esecuzione.")
            return
        self.reset_ui()

if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = PiCalculatorWindow()
    window.show()
    sys.exit(app.exec())
