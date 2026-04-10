#include "picalculatorwindow.h"
#include "calc.h"
#include "mem.h"

#include <QApplication>
#include <QComboBox>
#include <QFontMetrics>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QScreen>
#include <QTimer>

#include <chrono>
#include <stdexcept>
#include <thread>

using clock_type = std::chrono::steady_clock;

namespace {

    constexpr std::size_t LEIBNIZ_ITERATIONS_PARAL = 8000'000'000ULL;
    constexpr std::size_t LEIBNIZ_ITERATIONS = 8000'000'000ULL;
    constexpr std::size_t EULER_ITERATIONS = 8000'000'000ULL;
    constexpr std::size_t EULER_ITERATIONS_PARAL = 8000'000'000ULL;
    constexpr double RIEMANN_GAUSS_LIMIT = 1000.0;
    constexpr double RIEMANN_STEP = 1e-07;
    constexpr int FBELLARD_ITERATIONS = 100'000'000;
    constexpr int FBELLARD_ITERATIONS_PARAL = 1000'000'000;
    constexpr int WALLIS_ITERATIONS = 100'000'000;
    constexpr int WALLIS_ITERATIONS_PARAL = 1000'000'000;
    
} // namespace

double seconds_between(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

PiCalculatorWindow::PiCalculatorWindow(QWidget *parent)
    : QWidget(parent),
      title_("Calcolo di π con vari metodi"),
      algo_choices_({"Leibniz", "Euler", "F. Bellard", "Gaussian Integral", "Wallis", "MemTest"}),
      engine_choices_({"Single Core", "Multi Core"}) {
    init_ui();

    queue_timer_ = new QTimer(this);
    connect(queue_timer_, &QTimer::timeout, this, &PiCalculatorWindow::check_worker_result);
    queue_timer_->start(100);
}

PiCalculatorWindow::~PiCalculatorWindow() {
    if (queue_timer_ != nullptr) {
        queue_timer_->stop();
    }
    join_worker_if_needed();
}

void PiCalculatorWindow::reset_ui() {
//    engine_choice_->setEnabled(false);
//    engine_choice_->setCurrentIndex(0);
}

void PiCalculatorWindow::init_ui() {
    setWindowTitle(title_);

    auto *layout = new QGridLayout();
    setLayout(layout);

    label_copyright_ = new QLabel("(C) Alvise Dorigo");

    algo_choice_ = new QComboBox();
    algo_choice_->addItems(algo_choices_);
    connect(algo_choice_, &QComboBox::currentTextChanged, this, &PiCalculatorWindow::on_algo_choice);

    engine_choice_ = new QComboBox();
    engine_choice_->addItems(engine_choices_);

    calculate_button_ = new QPushButton("Calcola");
    connect(calculate_button_, &QPushButton::clicked, this, &PiCalculatorWindow::on_calculate_button_click);

    reset_button_ = new QPushButton("Reset");
    connect(reset_button_, &QPushButton::clicked, this, &PiCalculatorWindow::on_reset_button_click);

    result_label_ = new QLabel("");
    timer_label_ = new QLabel("");

    layout->addWidget(algo_choice_, 0, 0, 1, 2);
    layout->addWidget(engine_choice_, 1, 0, 1, 2, Qt::AlignLeft);
    layout->addWidget(calculate_button_, 2, 0, 1, 1);
    layout->addWidget(reset_button_, 2, 1, 1, 1);
    layout->addWidget(result_label_, 3, 0, 1, 2);
    layout->addWidget(timer_label_, 4, 0, 1, 2);
    layout->addWidget(label_copyright_, 99, 0, 1, 2, Qt::AlignRight | Qt::AlignBottom);

    layout->setRowStretch(98, 1);
    layout->setColumnStretch(1, 1);
    layout->setRowStretch(4, 1);

    algo_choice_->setCurrentIndex(0);
    on_algo_choice(algo_choice_->currentText());

    resize_and_center();
    setMinimumSize(350, 300);
    result_label_->setText("Risultato: ---");
    timer_label_->setText("Durata: ---");

    reset_ui();
}

void PiCalculatorWindow::resize_and_center() {
    const QFontMetrics font_metrics(font());
    const int title_width = font_metrics.horizontalAdvance(title_) + 5;

    QScreen *screen = QApplication::primaryScreen();
    if (screen == nullptr) {
        resize(350, 250);
        return;
    }

    const QRect screen_geometry = screen->availableGeometry();
    const int window_width = std::max(320, title_width);
    const int window_height = 250;

    const int center_x = screen_geometry.x() + (screen_geometry.width() - window_width) / 2;
    const int center_y = screen_geometry.y() + (screen_geometry.height() - window_height) / 2;

    setGeometry(center_x, center_y, window_width, window_height);
}

void PiCalculatorWindow::show_warning(const QString &title, const QString &text) {
    QMessageBox::warning(this, title, text);
}

void PiCalculatorWindow::set_ui_busy(bool busy) {
    calculate_button_->setEnabled(!busy);
    reset_button_->setEnabled(!busy);
    algo_choice_->setEnabled(!busy);
    //engine_choice_->setEnabled(!busy && algo_choice_->currentText() == "Gaussian Integral");
}

bool PiCalculatorWindow::is_worker_running() const {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    return worker_running_;
}

void PiCalculatorWindow::join_worker_if_needed() {
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void PiCalculatorWindow::worker_calculation(QString algorithm, QString engine) {
    WorkerMessage message;

    try {
        if (!algo_choices_.contains(algorithm)) {
            throw std::invalid_argument(QString("Algoritmo non supportato: %1").arg(algorithm).toStdString());
        }

        double result = 0.0;
        bool omp = false;
        if (engine == "Multi Core")
            omp = true;
        const auto start_time = std::chrono::steady_clock::now();
        
        //------------------------------------------------------------------
        if (algorithm == "Leibniz"){
            if(omp) {
                result = calc::pi_leibniz_omp(LEIBNIZ_ITERATIONS_PARAL);
            } else {
                result = calc::pi_leibniz(LEIBNIZ_ITERATIONS);
            }
        }

        //------------------------------------------------------------------
        if (algorithm == "Euler") {
            if(omp) {
                result = calc::pi_euler_omp(EULER_ITERATIONS_PARAL);
            } else {
                result = calc::pi_euler(EULER_ITERATIONS);
            }
        }

        //------------------------------------------------------------------
        if (algorithm == "F. Bellard") {
            result = calc::pi_fabrice_bellard(FBELLARD_ITERATIONS);
        }
        
        //------------------------------------------------------------------
        if (algorithm == "Gaussian Integral") {
            if(omp) {
                const auto iterations = static_cast<std::size_t>(RIEMANN_GAUSS_LIMIT / RIEMANN_STEP);
                result = calc::gaussian_integral_omp(iterations, RIEMANN_STEP);
            } else {
                const auto iterations = static_cast<std::size_t>(RIEMANN_GAUSS_LIMIT / RIEMANN_STEP);
                result = calc::gaussian_integral(iterations, RIEMANN_STEP);
            }
        }

        //------------------------------------------------------------------
        if (algorithm == "Wallis") {
            if(omp) {
                result = calc::wallis_omp(WALLIS_ITERATIONS_PARAL);
            } else {
                result = calc::wallis(WALLIS_ITERATIONS);
            }
        }

        //------------------------------------------------------------------
        if (algorithm == "MemTest") {
            const std::size_t buffer_size = 5ull * 1024ull * 1024ull * 1024ull;
            std::vector<char> buf(buffer_size);
            mem::mem_test_init( buf );
            auto t0 = clock_type::now();
            mem::mem_test_write( buf, 10 );
            auto t1 = clock_type::now();
            double write_seconds = seconds_between(t0, t1);
            double total_written = static_cast<double>(buffer_size) * 10;
            double write_gbs = total_written / write_seconds / 1e9;
            result = write_gbs;
        }
        
        const auto end_time = std::chrono::steady_clock::now();
        message.ok = true;
        message.result = result;
        message.duration = std::chrono::duration<double>(end_time - start_time).count();
    } catch (const std::exception &e) {
        message.ok = false;
        message.error = QString::fromUtf8(e.what());
    } catch (...) {
        message.ok = false;
        message.error = "Errore sconosciuto";
    }

    std::lock_guard<std::mutex> lock(worker_mutex_);
    pending_message_ = message;
    worker_running_ = false;
}

void PiCalculatorWindow::on_calculate_button_click() {
    if (is_worker_running()) {
        show_warning("Attenzione", "Un calcolo è già in esecuzione.");
        return;
    }

    join_worker_if_needed();

    const QString engine = engine_choice_->currentText();
    const QString algorithm = algo_choice_->currentText();

    result_label_->setText("Risultato: calcolo in corso...");
    timer_label_->setText("Durata: ...");
    set_ui_busy(true);

    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        pending_message_.reset();
        worker_running_ = true;
    }

    worker_thread_ = std::thread(&PiCalculatorWindow::worker_calculation, this, algorithm, engine);
}

void PiCalculatorWindow::check_worker_result() {
    std::optional<WorkerMessage> message;
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (!pending_message_.has_value()) {
            return;
        }
        message = std::move(pending_message_);
        pending_message_.reset();
    }

    join_worker_if_needed();

    if (message->ok) {
        result_label_->setText(QString("Risultato: %1").arg(message->result, 0, 'f', 8));
        if (message->duration < 1.0) {
            timer_label_->setText(QString("Durata: %1 secondi").arg(message->duration, 0, 'f', 6));
        } else {
            timer_label_->setText(QString("Durata: %1 secondi").arg(message->duration, 0, 'f', 3));
        }
    } else {
        result_label_->setText("Errore");
        timer_label_->setText("Durata: ---");
        show_warning("Errore", message->error);
    }

    set_ui_busy(false);
}

void PiCalculatorWindow::on_reset_button_click() {
    if (is_worker_running()) {
        show_warning("Attenzione", "Attendi la fine del calcolo prima di fare reset.");
        return;
    }

    join_worker_if_needed();
    result_label_->setText("Risultato: ---");
    timer_label_->setText("Durata: ---");
    algo_choice_->setCurrentIndex(0);
    reset_ui();
}

void PiCalculatorWindow::on_algo_choice(const QString &text) {
    engine_choice_->setDisabled(text == "F. Bellard");
}
