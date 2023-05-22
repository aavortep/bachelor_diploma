# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt6 UI code generator 6.3.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(310, 40, 211, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.loadButton.setFont(font)
        self.loadButton.setObjectName("loadButton")
        self.genreBox = QtWidgets.QComboBox(self.centralwidget)
        self.genreBox.setGeometry(QtCore.QRect(390, 110, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.genreBox.setFont(font)
        self.genreBox.setEditable(False)
        self.genreBox.setCurrentText("")
        self.genreBox.setObjectName("genreBox")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 110, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.tempoButton = QtWidgets.QPushButton(self.centralwidget)
        self.tempoButton.setGeometry(QtCore.QRect(110, 180, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.tempoButton.setFont(font)
        self.tempoButton.setStyleSheet("background-color: rgb(165, 216, 255);")
        self.tempoButton.setObjectName("tempoButton")
        self.measureButton = QtWidgets.QPushButton(self.centralwidget)
        self.measureButton.setGeometry(QtCore.QRect(470, 180, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.measureButton.setFont(font)
        self.measureButton.setStyleSheet("background-color: rgb(121, 243, 180);")
        self.measureButton.setObjectName("measureButton")
        self.tempoTable = QtWidgets.QTableWidget(self.centralwidget)
        self.tempoTable.setGeometry(QtCore.QRect(110, 240, 251, 301))
        self.tempoTable.setObjectName("tempoTable")
        self.tempoTable.setColumnCount(0)
        self.tempoTable.setRowCount(0)
        self.measureTable = QtWidgets.QTableWidget(self.centralwidget)
        self.measureTable.setGeometry(QtCore.QRect(470, 240, 251, 301))
        self.measureTable.setObjectName("measureTable")
        self.measureTable.setColumnCount(0)
        self.measureTable.setRowCount(0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.genreBox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Загрузить аудио (.mp3)"))
        self.label.setText(_translate("MainWindow", "Жанр:"))
        self.tempoButton.setText(_translate("MainWindow", "Определить темп"))
        self.measureButton.setText(_translate("MainWindow", "Определить размер"))
