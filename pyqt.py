import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from predictqt import test

class QFileDialogDemo(QWidget):
    def __init__(self):
        super(QFileDialogDemo,self).__init__()
        self.initUI()

    def initUI(self):
        gLayout=QVBoxLayout()
        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()

        self.resize(1200, 600)

        self.fname=''

        self.button1 = QPushButton('加载图片')
        self.button1.setFixedSize(150,30)
        self.button1.clicked.connect(self.loadImage)
        layout1.addWidget(self.button1)

        self.button2 = QPushButton('检测')
        self.button2.setFixedSize(150, 30)
        self.button2.clicked.connect(self.testIm)
        layout1.addWidget(self.button2)

        self.lineEdit = QLineEdit()
        self.lineEdit.setFixedSize(150,30)
        layout1.addWidget(self.lineEdit)


        self.imageLabel1 = QLabel()
        self.imageLabel1.setFixedSize(500,500)
        layout2.addWidget(self.imageLabel1)

        self.imageLabel2 = QLabel()
        self.imageLabel2.setFixedSize(500, 500)
        layout2.addWidget(self.imageLabel2)

        gLayout.addLayout(layout1)
        gLayout.addLayout(layout2)
        self.setLayout(gLayout)

        self.setWindowTitle('行人检测')


    def loadImage(self):
        self.fname,_ = QFileDialog.getOpenFileName(self,'打开文件','.','图像文件(*.jpg *.png)')
        print(self.fname)
        self.imageLabel1.setPixmap(QPixmap(self.fname))
        self.imageLabel1.setScaledContents(True)

    def testIm(self):

        num=test(self.fname,'test.jpg')
        self.imageLabel2.setPixmap(QPixmap('D:/file/pyproject/faceboxes/picture/test.jpg'))
        self.imageLabel2.setScaledContents(True)
        self.lineEdit.setText('人数：'+str(num))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QFileDialogDemo()
    main.show()
    sys.exit(app.exec_())