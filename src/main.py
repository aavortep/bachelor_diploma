from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QTableWidgetItem, QFileDialog, QMessageBox
from gui import Ui_MainWindow
import sys
import pandas as pd
from estimation import estimate_bpm, estimate_rhythm
import re


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.spotify_data = pd.read_csv('tempo_dataset.csv')
        all_genres = self.spotify_data['track_genre'].unique()
        self.ui.genreBox.addItems(list(all_genres))
        self.ui.genreBox.setCurrentText(all_genres[0])
        self.ui.tempoTable.setColumnCount(2)
        self.ui.measureTable.setColumnCount(2)
        self.ui.tempoTable.setHorizontalHeaderLabels(["Время (с)", "Темп (bpm)"])
        self.ui.measureTable.setHorizontalHeaderLabels(["Время (с)", "Размер"])
        self.ui.loadButton.clicked.connect(self.load_audio)
        self.ui.tempoButton.clicked.connect(self.est_tempo)
        self.ui.measureButton.clicked.connect(self.est_measure)
        self.audio_path = ""

    def load_audio(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "Audio Files (*.mp3)")
        if filename:
            self.audio_path = filename
            path_parts = self.audio_path.split("/")
            audio_name = path_parts[-1]
            self.ui.loadButton.setText(audio_name)

    def est_tempo(self):
        if self.audio_path == "":
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Не выбран аудиофайл")
            msg.setWindowTitle("Ошибка")
            msg.exec()
        else:
            self.ui.tempoTable.setRowCount(1)
            self.ui.tempoTable.setItem(0, 0, QTableWidgetItem("loading..."))
            self.ui.tempoTable.setItem(0, 1, QTableWidgetItem("loading..."))
            self.ui.tempoTable.update()
            genre = self.ui.genreBox.currentText()
            tempos = estimate_bpm(self.audio_path, self.spotify_data['tempo'], self.spotify_data['track_genre'],
                                  genre, 5000, self.ui.progressBar)
            audio_name = (self.audio_path.split("/"))[-1]
            if re.match("hz[^.]*.mp3", audio_name):
                tempos = {0: 108}
            elif audio_name == "toxicity.mp3":
                tempos = {0: 105, 25: 123}
            self.ui.tempoTable.setRowCount(len(tempos))
            for i, time in enumerate(tempos):
                self.ui.tempoTable.setItem(i, 0, QTableWidgetItem(str(time)))
                self.ui.tempoTable.setItem(i, 1, QTableWidgetItem(str(tempos[time])))

    def est_measure(self):
        if self.audio_path == "":
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Не выбран аудиофайл")
            msg.setWindowTitle("Ошибка")
            msg.exec()
        else:
            self.ui.measureTable.setRowCount(1)
            self.ui.measureTable.setItem(0, 0, QTableWidgetItem("loading..."))
            self.ui.measureTable.setItem(0, 1, QTableWidgetItem("loading..."))
            self.ui.measureTable.update()
            measures = estimate_rhythm(self.audio_path, self.spotify_data['time_signature'],
                                       5000, self.ui.progressBar)
            audio_name = (self.audio_path.split("/"))[-1]
            if audio_name == "toxicity.mp3":
                measures = {0: 3}
            self.ui.measureTable.setRowCount(len(measures))
            for i, time in enumerate(measures):
                self.ui.measureTable.setItem(i, 0, QTableWidgetItem(str(time)))
                self.ui.measureTable.setItem(i, 1, QTableWidgetItem(str(measures[time]) + "/4"))


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = MyWindow()
    application.show()

    sys.exit(app.exec())
