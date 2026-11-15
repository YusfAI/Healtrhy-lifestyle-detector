"""Healthy Lifestyle Detector: desktop app (PySide6).

Answer 8 lifestyle questions; a Random Forest trained on the "Food choices"
survey estimates the probability that your diet is healthy / balanced and shows
which habits weigh most in the model.

Run:  python app.py
"""

import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from model import FEATURES, TrainedModel, train

# Probability above which the profile is labelled "healthy".
THRESHOLD = 0.5


class TrainingThread(QThread):
    """Trains and cross-validates the model without freezing the window."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.finished_ok.emit(train())
        except FileNotFoundError:
            self.failed.emit("File 'food_choices.csv' not found next to app.py.")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc))


class LifestyleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Healthy Lifestyle Detector")
        self.setGeometry(100, 100, 680, 820)
        self.model: TrainedModel | None = None
        self.inputs: dict[str, QComboBox] = {}

        self.setup_ui()
        self.start_training()

    # ---------- UI ----------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("🍎 Healthy Lifestyle Detector")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22pt; font-weight: bold; padding: 16px; color: #2E7D32;")
        layout.addWidget(title)

        form_group = QGroupBox("Your habits")
        form = QFormLayout()
        for feature, (question, answers) in FEATURES.items():
            combo = QComboBox()
            for code, label in answers.items():
                combo.addItem(label, code)
            form.addRow(QLabel(question), combo)
            self.inputs[feature] = combo
        form_group.setLayout(form)
        layout.addWidget(form_group)

        self.btn_predict = QPushButton("🔮 Analyse my lifestyle")
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet(
            """
            QPushButton { background-color: #4CAF50; color: white; padding: 14px;
                          font-size: 13pt; font-weight: bold; border-radius: 8px; }
            QPushButton:hover:enabled { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
            """
        )
        self.btn_predict.clicked.connect(self.predict)
        layout.addWidget(self.btn_predict)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(260)
        self.show_message("⚙️ Training and cross-validating the model…", "#2196F3")
        layout.addWidget(self.result)

    def show_message(self, text: str, color: str):
        self.result.setHtml(
            f"<p style='color:{color}; text-align:center; font-size:12pt; padding:24px;'>{text}</p>"
        )

    # ---------- Model ----------
    def start_training(self):
        self.training_thread = TrainingThread()
        self.training_thread.finished_ok.connect(self.on_trained)
        self.training_thread.failed.connect(lambda msg: self.show_message(f"❌ {msg}", "red"))
        self.training_thread.start()

    def on_trained(self, model: TrainedModel):
        self.model = model
        self.btn_predict.setEnabled(True)
        m = model.metrics
        self.show_message(
            f"✅ Model ready (Random Forest, {model.n_samples} survey answers)<br>"
            f"Cross-validated balanced accuracy: {m['balanced_accuracy']:.0%} · "
            f"ROC AUC: {m['roc_auc']:.2f}",
            "green",
        )

    def current_answers(self) -> dict[str, int]:
        return {feature: combo.currentData() for feature, combo in self.inputs.items()}

    def predict(self):
        if not self.model:
            return
        proba = self.model.predict_proba(self.current_answers())
        healthy = proba >= THRESHOLD
        color = "#4CAF50" if healthy else "#F44336"
        verdict = "✅ Healthy lifestyle" if healthy else "⚠️ Room for improvement"

        top = list(self.model.importances.items())[:3]
        drivers = "".join(
            f"<li>{FEATURES[name][0]} <span style='color:#888'>({weight:.0%})</span></li>"
            for name, weight in top
        )

        self.result.setHtml(
            f"""
            <div style='text-align:center; padding:10px;'>
              <h1 style='color:{color}; margin:10px;'>{verdict}</h1>
              <p style='font-size:14pt;'>Probability of a healthy / balanced diet:
                 <b>{proba:.0%}</b></p>
            </div>
            <div style='padding:0 16px; font-size:10pt; color:#444;'>
              <b>Habits that weigh most in the model:</b>
              <ul>{drivers}</ul>
              <p style='color:#888;'>Trained on {self.model.n_samples} student survey answers;
              cross-validated balanced accuracy
              {self.model.metrics['balanced_accuracy']:.0%}. Educational project, not medical advice.</p>
            </div>
            """
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LifestyleApp()
    window.show()
    sys.exit(app.exec())
