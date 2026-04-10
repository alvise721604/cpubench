#pragma once

#include <QWidget>

#include <mutex>
#include <optional>
#include <thread>

class QLabel;
class QPushButton;
class QComboBox;
class QTimer;

class Window : public QWidget {
    Q_OBJECT

public:
    explicit Window(QWidget *parent = nullptr);
    ~Window() override;

private slots:
    void on_algo_choice(const QString &text);
    void on_calculate_button_click();
    void on_reset_button_click();
    void check_worker_result();

private:
    struct WorkerMessage {
        bool ok = false;
        double result = 0.0;
        double duration = 0.0;
        QString error;
    };

    void init_ui();
    void reset_ui();
    void resize_and_center();
    void show_warning(const QString &title, const QString &text);
    void set_ui_busy(bool busy);
    void worker_calculation(QString algorithm, QString engine);
    bool is_worker_running() const;
    void join_worker_if_needed();

    QString title_;
    QStringList algo_choices_;
    QStringList engine_choices_;

    QLabel *label_copyright_ = nullptr;
    QComboBox *algo_choice_ = nullptr;
    QComboBox *engine_choice_ = nullptr;
    QPushButton *calculate_button_ = nullptr;
    QPushButton *reset_button_ = nullptr;
    QLabel *result_label_ = nullptr;
    QLabel *timer_label_ = nullptr;
    QTimer *queue_timer_ = nullptr;

    std::thread worker_thread_;
    mutable std::mutex worker_mutex_;
    bool worker_running_ = false;
    std::optional<WorkerMessage> pending_message_;
};
