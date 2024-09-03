import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer
import cv2
from fer import FER

class EmotionalMirror(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotional Mirror")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #2E3440; color: #D8DEE9;")

        # لیبل برای نمایش ویدیو
        self.image_label = QLabel(self)
        self.image_label.setGeometry(150, 50, 700, 500)
        self.image_label.setStyleSheet("border: 3px solid #81A1C1;")

        # قسمت پیام انگیزشی
        self.message_label = QLabel(self)
        self.message_label.setGeometry(150, 580, 700, 50)
        self.message_label.setFont(QFont('Arial', 14))
        self.message_label.setStyleSheet("color: #88C0D0; border: 2px solid #81A1C1; padding: 10px;")

        # راه‌اندازی دوربین
        self.cap = cv2.VideoCapture(0)

        # تنظیم تایمر برای به‌روزرسانی فریم‌ها
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # بارگذاری مدل تشخیص احساسات
        self.detector = FER(mtcnn=True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # تشخیص احساسات
            result = self.detector.detect_emotions(frame)
            if result:
                bounding_box = result[0]["box"]
                emotions = result[0]["emotions"]
                max_emotion = max(emotions, key=emotions.get)

                # رسم مستطیل دور چهره و نمایش احساس
                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (129, 204, 216), 2)
                cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (129, 204, 216), 2, cv2.LINE_AA)

                # نمایش پیام انگیزشی براساس احساس
                self.display_message(max_emotion)

            # تبدیل تصویر به فرمت مناسب برای PyQt
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def display_message(self, emotion):
        messages = {
            "angry": "Take a deep breath and stay calm.",
            "disgust": "Remember, every experience is a lesson.",
            "fear": "Face your fears with courage.",
            "happy": "Keep smiling! The world needs your positivity.",
            "sad": "It's okay to feel sad sometimes. Everything will be better.",
            "surprise": "Embrace the unexpected moments of life.",
            "neutral": "Stay balanced and centered."
        }
        message = messages.get(emotion, "Stay positive!")
        self.message_label.setText(f"Detected Emotion: {emotion.capitalize()} - {message}")

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionalMirror()
    window.show()
    sys.exit(app.exec_())
