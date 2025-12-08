import sys
import pandas as pd
import warnings
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QFormLayout, QGroupBox, QLabel, QPushButton,
                               QComboBox, QSpinBox, QTextEdit)
from PySide6.QtCore import Qt, QThread, Signal
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


class TrainingThread(QThread):
    """Background thread for model training"""
    finished = Signal(object, float)
    error = Signal(str)

    def __init__(self, df):
        super().__init__()
        self.df = df

    def run(self):
        try:
            # Prepare data
            features = ['diet_current', 'fruit_day', 'veggies_day', 'exercise',
                        'healthy_feeling', 'calories_day']
            X = self.df[features].copy()
            y = self.df['healthy_meal'].apply(
                lambda v: 1 if any(w in str(v).lower()
                                   for w in ['healthy', 'yes', 'good', 'balanced']) else 0
            )

            # Simple preprocessing - convert to numeric codes
            X['diet_code'] = X['diet_current'].map({
                'Vegan': 3, 'Vegetarian': 2, 'None': 0
            }).fillna(1)

            X['exercise_code'] = X['exercise'].map({
                'Daily': 2, 'Weekly': 1, 'None': 0
            }).fillna(0)

            X['feeling_code'] = X['healthy_feeling'].map({
                'Yes': 1, 'No': 0
            }).fillna(0)

            X['fruit_day'] = pd.to_numeric(X['fruit_day'], errors='coerce').fillna(2)
            X['veggies_day'] = pd.to_numeric(X['veggies_day'], errors='coerce').fillna(3)
            X['calories_day'] = pd.to_numeric(X['calories_day'], errors='coerce').fillna(2000)

            # Use only numeric features
            feature_cols = ['diet_code', 'exercise_code', 'feeling_code',
                            'fruit_day', 'veggies_day', 'calories_day']
            X_clean = X[feature_cols]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.3, random_state=42, stratify=y
            )

            # Simple Decision Tree with balanced classes
            model = DecisionTreeClassifier(
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',  # CRITICAL: Handles imbalanced data
                min_impurity_decrease=0.001,
                random_state=42
            )

            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

            # Debug: Print class distribution
            print(f"Training - Healthy: {sum(y_train == 1)}, Unhealthy: {sum(y_train == 0)}")

            self.finished.emit(model, accuracy)

        except Exception as e:
            self.error.emit(str(e))


class MealPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Healthy Meal Predictor")
        self.setGeometry(100, 100, 650, 700)

        self.model = None
        self.accuracy = 0

        self.setup_ui()
        self.load_and_train()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("🍎 Healthy Meal Predictor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24pt; font-weight: bold; padding: 20px; color: #2E7D32;")
        layout.addWidget(title)

        # Form
        form_group = QGroupBox("Enter Your Information")
        form_layout = QFormLayout()

        self.combo_diet = QComboBox()
        self.combo_diet.addItems(["None", "Vegetarian", "Vegan", "Other"])
        form_layout.addRow("Diet Type:", self.combo_diet)

        self.combo_exercise = QComboBox()
        self.combo_exercise.addItems(["None", "Weekly", "Daily"])
        form_layout.addRow("Exercise:", self.combo_exercise)

        self.combo_feeling = QComboBox()
        self.combo_feeling.addItems(["No", "Yes"])
        form_layout.addRow("Feel Healthy?", self.combo_feeling)

        self.spin_fruit = QSpinBox()
        self.spin_fruit.setMaximum(20)
        self.spin_fruit.setValue(2)
        form_layout.addRow("Fruits/Day:", self.spin_fruit)

        self.spin_veggie = QSpinBox()
        self.spin_veggie.setMaximum(20)
        self.spin_veggie.setValue(3)
        form_layout.addRow("Veggies/Day:", self.spin_veggie)

        self.spin_calories = QSpinBox()
        self.spin_calories.setMinimum(500)
        self.spin_calories.setMaximum(5000)
        self.spin_calories.setSingleStep(100)
        self.spin_calories.setValue(2000)
        form_layout.addRow("Calories/Day:", self.spin_calories)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Button
        self.btn_predict = QPushButton("🔮 Predict")
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_predict.clicked.connect(self.predict_meal)
        layout.addWidget(self.btn_predict)

        # Result
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setMinimumHeight(200)
        self.text_result.setHtml(
            "<p style='color: #FF9800; text-align: center; padding: 30px;'>⏳ Loading model...</p>"
        )
        layout.addWidget(self.text_result)

    def load_and_train(self):
        try:
            df = pd.read_csv("food_choices.csv")
            self.text_result.setHtml(
                "<p style='color: #2196F3; text-align: center; padding: 30px;'>⚙️ Training model...</p>"
            )

            self.training_thread = TrainingThread(df)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            self.training_thread.start()

        except FileNotFoundError:
            self.text_result.setHtml("<p style='color: red;'>❌ File 'food_choices.csv' not found!</p>")
        except Exception as e:
            self.text_result.setHtml(f"<p style='color: red;'>❌ Error: {str(e)}</p>")

    def on_training_finished(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
        self.btn_predict.setEnabled(True)
        self.text_result.setHtml(
            f"<p style='color: green; text-align: center; font-size: 12pt; padding: 20px;'>"
            f"✅ Model Ready!<br>Decision Tree Trained<br>Accuracy: {accuracy:.1%}</p>"
        )

    def on_training_error(self, error_msg):
        self.text_result.setHtml(f"<p style='color: red;'>❌ Error:<br>{error_msg}</p>")

    def prepare_input(self):
        """Convert user input to model format"""
        diet_map = {'Vegan': 3, 'Vegetarian': 2, 'None': 0, 'Other': 1}
        exercise_map = {'Daily': 2, 'Weekly': 1, 'None': 0}
        feeling_map = {'Yes': 1, 'No': 0}

        return pd.DataFrame([{
            'diet_code': diet_map.get(self.combo_diet.currentText(), 1),
            'exercise_code': exercise_map[self.combo_exercise.currentText()],
            'feeling_code': feeling_map[self.combo_feeling.currentText()],
            'fruit_day': float(self.spin_fruit.value()),
            'veggies_day': float(self.spin_veggie.value()),
            'calories_day': float(self.spin_calories.value())
        }])

    def predict_meal(self):
        if not self.model:
            return

        try:
            # Get prediction
            user_input = self.prepare_input()
            probabilities = self.model.predict_proba(user_input)[0]

            # Use lower threshold for "healthy" prediction
            threshold = 0.35
            is_healthy = (probabilities[1] >= threshold)
            confidence = probabilities[1] if is_healthy else probabilities[0]

            # Result
            result_text = "Healthy" if is_healthy else "Unhealthy"
            color = "#4CAF50" if is_healthy else "#F44336"
            emoji = "✅" if is_healthy else "❌"

            # Get user values for display
            values = user_input.iloc[0].to_dict()

            self.text_result.setHtml(f"""
            <div style='text-align: center; padding: 15px;'>
                <h1 style='color: {color}; margin: 15px;'>{emoji} {result_text}</h1>

                <div style='background-color: #f5f5f5; border-radius: 8px; padding: 15px; margin: 15px;'>
                    <p style='font-size: 14pt;'>
                        <b>Confidence:</b> {confidence:.0%}
                    </p>
                    <p style='font-size: 11pt; color: #666;'>
                        Healthy probability: {probabilities[1]:.1%}<br>
                        Model Accuracy: {self.accuracy:.1%}
                    </p>
                </div>

                <div style='border-top: 2px solid #ddd; padding-top: 15px; margin-top: 15px;'>
                    <p style='font-size: 10pt; color: #555;'>
                        <b>Your Input:</b><br>
                        {self.combo_diet.currentText()} diet • {self.combo_exercise.currentText()} exercise<br>
                        {values['fruit_day']:.0f} fruits, {values['veggies_day']:.0f} veggies<br>
                        {values['calories_day']:.0f} calories • Feel healthy: {self.combo_feeling.currentText()}
                    </p>
                </div>
            </div>
            """)

        except Exception as e:
            self.text_result.setHtml(f"<p style='color: red;'>❌ Error: {str(e)}</p>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MealPredictorApp()
    window.show()
    sys.exit(app.exec())
