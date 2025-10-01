# By Melody and Johnathan
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from torchvision import datasets, models, transforms
import torch.nn as nn
# import models
from PyQt5.QtGui import QPalette, QBrush, QPixmap
global img
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torchvision
import os
import argparse

def get_args_parser(log_dir, resume):
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')  #############
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--log_dir', default=log_dir,
                        help='path where to tensorboard log')  ##############
    parser.add_argument('--resume', default=resume,
                        # parser.add_argument('--resume', default='',
                        help='resume from checkpoint')  ###########
    return parser


def main(test_image_path='', classes=None):
    if classes == "bird":
        class_dict = {
                'Non_Recyclable_Materials': 0,'Organic_Waste': 1, 'Recyclable_Materials': 2
            }######
        test_valid_transforms = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])])
        resnet50 = torchvision.models.resnet50(pretrained=False)
        for param in resnet50.parameters():
            param.requires_grad = False
        fc_inputs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3),####
            nn.LogSoftmax(dim=1)
        )
  
        PATH="/Users/melodyzi/Documents/BostonU/Spring_2024/TRASHIMAGE/trained_models/my-10_model_16.pth"
        resnet50.load_state_dict(torch.load(PATH,map_location='cpu'))
        resnet50.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_image_path = test_image_path
        image = Image.open(test_image_path)
        image_tensor = test_valid_transforms(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        resnet50 = resnet50.to(device)
        outputs = resnet50(image_tensor)
        ret, predictions = torch.max(outputs.data, 1)
        probabilities = F.softmax(outputs, dim=1)
        max_prob1, _ = torch.max(probabilities, dim=1)
        print("ret",max_prob1.item())
        value =predictions.item()

        print(list(class_dict.keys())[list(class_dict.values()).index(value)])
        content = test_image_path + "+" + list(class_dict.keys())[list(class_dict.values()).index(value)] + " "+ str(round(max_prob1.item(),2))
        print(content)
 
        file_path = "detect.txt"

        with open(file_path, 'w') as file:
            file.write(content)


# from deeplab import DeeplabV3
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):  
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 800)
        self.MainWindow = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(910, 10, 100, 50))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 10, 100, 50))
        self.pushButton_4.setObjectName("pushButton_4")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(510, 10, 100, 50))
        self.pushButton_5.setObjectName("pushButton_5")

        self.label1 = QtWidgets.QTextBrowser(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(140, 10, 200, 50))
        self.label1.setStyleSheet("background-color: lightgray;")

        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(150, 80, 800, 800))
        self.label2.setObjectName("label2")
        self.label3 = QtWidgets.QLabel(self.centralwidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1128, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_3.clicked.connect(self.getback)
        self.pushButton_4.clicked.connect(self.detect)
        self.pushButton_5.clicked.connect(self.car)
        self.image_path = None
        self.num = 0
        self.classes = None

    def retranslateUi(self, MainWindow):  
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_3.setText(_translate("MainWindow", "drop out"))
        self.pushButton_4.setText(_translate("MainWindow", "detect"))
        self.pushButton_5.setText(_translate("MainWindow", "upload"))
        self.pushButton_3.setStyleSheet("color: SimHei; font-size: 15px;")
        self.pushButton_4.setStyleSheet("color: SimHei; font-size: 15px;")
        self.pushButton_5.setStyleSheet("color: SimHei; font-size: 15px;")

        self.label2.setText(_translate("MainWindow", ""))
        self.label3.setText(_translate("MainWindow", ""))


    def fruit(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        self.classes = "fruit"
        if image_path:
            pixmap = QtGui.QPixmap(image_path)
            self.label2.setPixmap(pixmap)
            self.label2.setScaledContents(True)

    def car(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        self.classes = "bird"
        if image_path:
            pixmap = QtGui.QPixmap(image_path)
            self.label2.setPixmap(pixmap)
            self.label2.setScaledContents(True)

    def getback(self):
        self.MainWindow.close()

    def detect(self):
        main(self.image_path, self.classes)
        file_path = "detect.txt"

        with open(file_path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
        print("last line:", last_line)
        t=last_line.split('+')
        self.label1.setText(t[1])
        self.label1.setStyleSheet("color: red; font-family: 'SimHei'; font-size: 20px; text-align: center;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()  
    ui = Ui_MainWindow()  
    palette = QPalette()
    ui.setupUi(MainWindow1)
    palette.setBrush(QPalette.Background, QBrush(
        QPixmap("background.png")))
    MainWindow1.setPalette(palette)
    MainWindow1.show()
    sys.exit(app.exec_())
