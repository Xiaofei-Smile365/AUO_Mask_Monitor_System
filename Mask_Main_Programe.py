# -*- Encoding: UTF-8 -*-

"""
@File        : AI Smart Version

@Contact     : xiaofei.smile365@Gmail.com
@Author      : 苏晓飞
@Call        ：8690-2484/0070
@IDE         : PyCharm

@Version     : 3.0
@Modify Time : 5/24/20 08:01 AM
@友达光电（苏州）有限公司 AUSZ-S06
"""

"""
# 对于无开始结束信号、循环检测NG目标类型的专案，以下仅三处需修改，均使用“#############”进行明显标识
# 第一部分为逻辑判断部分，检测目标的名称需根据实际情况修改，逻辑部分可根据实际情况修改或保持
# 第二部分为YOLO函数内的相关配置文件，务必修改为相对应的配置文件
# 第三部分为专案名称等配置参数，务必修改为相对应的实际参数
"""


import sys  # 载入系统sys函数模块
import os  # 载入系统os函数模块

if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']  # PyQt自身存在bug，打包时环境变量出错，无法运行，此语句对环境变量进行重新配置，消除bug
from PyQt5 import QtWidgets, QtGui, QtCore  # 载入PyQt相应的函数模块
from PyQt5.QtCore import Qt  # 同上，部分函数库可能存在重复
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPalette, QBrush  # 同上
from PyQt5.QtWidgets import *  # 在Qt5中使用的基本的GUI窗口控件都在PyQt5.QtWidgets模块中

import datetime  # 载入系统时间函数模块

import cv2  # 载入opencv函数模块，此程序中用于获取摄像头影像
import darknet  # 载入darknet模块，此模块隶属于darknet框架的python接口

sys.path.append('/opt/nvidia/jetson-gpio/lib/python/')  # 用于GPIO口的相关配置
sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')  # 同上

import Jetson.GPIO as GPIO  # 载入GPIO函数模块，并使用别名GPIO


def convertBack(x, y, w, h):  # 此函数用于darknet检测结果框的坐标转换，隶属于darknet框架的python接口；（x，y）为检测结果框的中心点坐标，w为检测框的宽度，h为检测框的高度
    xmin = int(round(x - (w / 2)))  # 此为结果框的左上角X坐标
    xmax = int(round(x + (w / 2)))  # 此为结果框的右下角X坐标
    ymin = int(round(y - (h / 2)))  # 此为结果框的左上角Y坐标
    ymax = int(round(y + (h / 2)))  # 此为结果框的右下角Y坐标
    return xmin, ymin, xmax, ymax  # 将计算结果返回


def clean_alarm(self):
    self.curr_value = GPIO.LOW  # 设定GPIO端口低电平
    GPIO.output(self.output_pin, self.curr_value)  # 对端口进行设置，self.output_pin为输出的针脚，默认设置为１１，self.curr_value为电平变量
    print("Outputting {} to pin {}".format(self.curr_value, self.output_pin))  # 在终端打印此条信息，便于程序调试，例：Outputting ０ to pin １１

    self.label_alarm.setPixmap(QPixmap('./sources_file/green.png'))  # 设定label内容为图片，实现红绿灯显示

def red(self):
    self.curr_value = GPIO.HIGH  # 设定GPIO端口高电平
    GPIO.output(self.output_pin, self.curr_value)  # 对端口进行设置，self.output_pin为输出的针脚，默认设置为１１，self.curr_value为电平变量
    print("Outputting {} to pin {}".format(self.curr_value, self.output_pin))  # 在终端打印此条信息，便于程序调试，例：Outputting １ to pin １１

    self.label_alarm.setPixmap(QPixmap('./sources_file/alarm.png'))  # 设定label内容为图片，实现红绿灯显示

    self.ng_sum = self.ng_sum + 1  # 将UI界面的NG数量加１
    self.lcd_ng.setText(str(self.ng_sum))  # 重新设定UI界面的NG Label，使其更新NG数量显示
    print("Now NG Sum is {}\n".format(self.ng_sum))  # 在终端打印此条消息，便于程序调试，例：Now NG Sum is ６

def cvDrawBoxes(self, detections, img):  # 此函数用于在每一帧图像上画出识别框；智眸系统的逻辑判断程序亦包含于此函数
    for detection in detections:  # 通过for循环画出所有检测到的识别框
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]  # detection为darknet返回的列表
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))  # 使用convertBack函数进行坐标转换
        pt1 = (xmin, ymin)  # pt1为识别框的左上角坐标
        pt2 = (xmax, ymax)  # pt２为识别框的右下角坐标
        
#############################################################################################################################
# 以下为智眸系统的逻辑判断部分，原理为通过判断每帧照片是否出现NG目标，并设定相应阀值，连续出现一定帧数的NG目标后触发报警并对UI界面的相应元素进行设定（NG数量、报警灯颜色等）
#############################################################################################################################
        if detection[0].decode() == 'No_Mask' or detection[0].decode() == 'Wear_Mask':  # 判断此帧照片是否存在NG目标，例如person此目标；实际运行时将NG修改为相应的目标名称；因识别框绘制程序位于if判断语句内，因此只有设定过的目标才会被绘制出识别框。对于官方模型，可以设别多种目标，但是因为if语句设定了目标名称为person是才绘制识别框，因为实时监控界面只可看到person目标的检测，从而实现只检测需要的目标（实际上其他目标仍然被检测出，只是未绘制识别框，依旧消耗了GPU资源）
        
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)  # 绘制识别框，通过修改此语句可实现绘制点的绘制或者其他图案；img为此帧图像，pt１为左上角坐标，pt２为右下角坐标，（０， ２５５， ０）为识别框的颜色（RGB模式）
            cv2.putText(img, detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)  # 绘制识别框上的文字信息，img为图像，detection[0].decode()为目标名称（NG或person），str(round(detection[1] * 100, 2))为置信度（detection【１】为置信度，２为保留两位小数），[0, 255, 0]为文字颜色
        if detection[0].decode() == 'No_Mask':
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 3)
            cv2.putText(img, detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)
        if detection[0].decode() == 'Wear_Mask':
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
            cv2.putText(img, detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)

        if detection[0].decode() == 'No_Mask':
            self.photo_ng_mark = 1  # 设定照片NG mark，此帧照片是否出现NG目标，如果出现则置１，否则维持初值０

    if self.photo_ng_mark == 0:  # 判断照片NG mark，等于０则代表照片无NG目标
        self.real_no_alarm_mark = self.real_no_alarm_mark + 1  # 对无NG目标的帧数进行累积

        if self.real_no_alarm_mark == 3:  # 当无NG目标的帧数累积到２４帧，可根据实际情况修改此阀值
            self.real_ng_mark = 0  # 对有NG目标的帧数进行清０

            clean_alarm(self)

    if self.photo_ng_mark == 1:  # 基本原理同上；判断照片NG mark，等于１则代表照片有NG目标
        self.real_ng_mark = self.real_ng_mark + 1  # 对有NG目标的帧数进行累积

        if self.real_ng_mark == 3:  # 当有NG目标的帧数累积到２４帧，可根据实际情况修改此阀值
            self.real_no_alarm_mark = 0  # 对无NG目标的帧数进行清０

            red(self)

            self.real_image = img  # 图像赋给另一变量，实现界面显示
            self.real_image = QtGui.QImage(self.real_image[:], self.real_image.shape[1], self.real_image.shape[0], self.real_image.shape[1] * 3, QtGui.QImage.Format_RGB888)  # 对图像格式进行转换
            if self.ng_image_site == 5:
                self.ng_image_site = 1
            if self.ng_image_site == 1:
                self.label_image_ng_1.setPixmap(QPixmap(self.real_image))  # 定义label的内容为图片，并显示初始图像
            if self.ng_image_site == 2:
                self.label_image_ng_2.setPixmap(QPixmap(self.real_image))  # 定义label的内容为图片，并显示初始图像
            if self.ng_image_site == 3:
                self.label_image_ng_3.setPixmap(QPixmap(self.real_image))  # 定义label的内容为图片，并显示初始图像
            if self.ng_image_site == 4:
                self.label_image_ng_4.setPixmap(QPixmap(self.real_image))  # 定义label的内容为图片，并显示初始图像
            self.ng_image_site = self.ng_image_site + 1

    self.photo_ng_mark = 0  # 将照片NG mark重置为０

    return img  # 将绘制过识别框＆文字的返回


netMain = None  # 此变量为darknet框架自带
metaMain = None  # 同上
altNames = None  # 同上


def YOLO(self):  # 此函数用于darknet所需文件的路径声明&打开摄像头等功能
    global metaMain, netMain, altNames  # darknet框架自带变量，将其声明为全局变量
#############################################################################################################################
# 更改以下三个配置文件&查看一个配置文件，重要！！！
#############################################################################################################################
    configPath = "./yolov3-tiny.cfg"  # 配置cfg文件的路径
    weightPath = "./yolov3-tiny_final.weights"  # 配置weight文件的路径
    metaPath = "./voc.data"  # 配置data文件的路径（切忌查看data文件内定义的name文件的路径&内容是否与实际相对应）
    
    if not os.path.exists(configPath):  # 判断cfg文件是否存在
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")  # 若不存在，则报错
    if not os.path.exists(weightPath):  # 判断weight文件是否存在
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")  # 若不存在，则报错
    if not os.path.exists(metaPath):  # 判断data文件是否存在
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")  # 若不存在，则报错
    if netMain is None:  # 判断netMain是否为None
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1 导入weight模型文件
    if metaMain is None:  # 判断netMain是否为None
        metaMain = darknet.load_meta(metaPath.encode("ascii"))  # 导入data配置文件
    if altNames is None:  # 判断netMain是否为None
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
        
    self.cap = cv2.VideoCapture("rtsp:192.168.1.211/user=admin&password=&channel=1&stream=0.sdp?")  # 设定USB摄像头端口为0，使用opencv打开摄像头
    if self.cap.isOpened() == 0:  # 判断USB摄像头是否被打开，如果未打开，则更换摄像头端口
        self.cap = cv2.VideoCapture(1)  # 更换USB摄像头端口为1
    self.cap.set(3, 768)  # 设定摄像头视频流尺寸
    self.cap.set(4, 1024)  # 设定摄像头视频流尺寸
    print("Starting the YOLO loop...")  # 在终端打印此消息，便于调试；此位置代表摄像头已被打开，即将进入YOLO检测环节
    self.camera_state = 1  # 设定摄像头状态为1，便于Log档记录

    # Create an image we reuse for each detect
    self.darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain), 3)
    self.camera_state = 1  # 设定摄像头状态为1，便于Log档记录


class MainWindow(QWidget):  # 创建UI界面的类
    def __init__(self, parent=None):  # 基础窗口控件QWidget类是所有用户界面对象的基类， 所有的窗口和控件都直接或间接继承自QWidget类。
        super(MainWindow, self).__init__(parent)  # 使用super函数初始化窗口

        self.resize(1920, 1080)  # 将UI界面初始尺寸设定为1920*1080
        self.setWindowTitle('智眸')  # 设定窗口控件的标题为 智眸
        self.setWindowIcon(QIcon('./sources_file/Video_AI.ico'))  # 设定窗口的图标

        palette = QPalette()  # 使用Qpalette函数进行UI界面北京的设置
        palette.setBrush(QPalette.Background, QBrush(QPixmap('./sources_file/blue_background.png').scaled(self.width(), self.height())))  # 设定UI界面背景图片
        self.setPalette(palette)  # 设定UI界面北京
        self.setAutoFillBackground(True)  # 设定背景图片自动全屏填充


        self.ui_addwidget()  # 定义UI界面的基本元素，并创建控件
        self.ui_addlayout_for_widget()  # 对控件进行布局，创建界面局部布局
        self.ui_addlayout_for_layout()  # 对局部布局进行布局，创建界面总体布局

    def ui_addwidget(self):  # 此函数内定义UI界面的基本元素，并创建控件

        self.label_title = QLabel(self)  # 定义标题label
        self.label_title.setText('<b>智眸-口罩佩戴实时检测系统<b>')  # 定义标题label的文字内容
        self.label_title.setAlignment(Qt.AlignCenter)  # 定义标题label文字内容的对齐方式（中心对齐）
        self.label_title.setStyleSheet('color: rgb(241, 172, 1)')  # 定义标题label文字内容的字体颜色
        self.label_title.setFont(QFont('SanSerif', 45))  # 定义标题label文字内容的字体大小
        self.label_title.setFixedSize(1000, 80)  # 定义标题label的尺寸大小
        self.h_box_title = QHBoxLayout()  # 创建水平布局
        self.h_box_title.addWidget(self.label_title)  # 将标题label添加到水平布局

        self.label_owner = QLabel(self)  # 定义label
        self.label_owner.setText('<b>艾聚达信息技术(苏州)有限公司<b>')  # 定义label的文字内容
        self.label_owner.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_owner.setStyleSheet('color: rgb(0, 205, 0)')  # 定义label文字内容的字体颜色
        self.label_owner.setFont(QFont('SanSerif', 16))  # 定义label文字内容的字体大小
        self.label_owner.setFixedSize(300, 30)  # 定义label的尺寸大小
        self.h_box_owner = QHBoxLayout()  # 创建水平布局
        self.h_box_owner.addWidget(self.label_owner)  # 将label添加到水平布局

        self.label_datetime = QLabel(self)  # 定义label
        self.label_datetime.setText('<b>1997/01/01 00:00:00<b>')  # 定义label的内容为图片，并显示初始图像
        self.label_datetime.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_datetime.setStyleSheet('color: rgb(0, 205, 0)')  # 定义label文字内容的字体颜色
        self.label_datetime.setFont(QFont('SanSerif', 16))  # 定义label文字内容的字体大小
        self.label_datetime.setFixedSize(250, 30)  # 定义label的尺寸大小
        self.real_time = QtCore.QTimer()  # 创建定时器（定时器类似线程的概念，1.另一线程执行link的函数 2.每隔一定时间执行一次，可实现类似try的概念，单次执行错误，不会退出）
        self.real_time.timeout.connect(self.date_time)  # 到时间后执行date_time函数
        self.real_time.start(1000)  # 每隔1000ms（1s）执行一次
        self.h_box_datetime = QHBoxLayout()  # 创建水平布局
        self.h_box_datetime.addWidget(self.label_datetime)  # 将label添加到水平布局

        self.label_image = QLabel(self)  # 定义label
        self.label_image.setPixmap(QPixmap('./sources_file/image_sample.png'))  # 定义label的内容为图片，并显示初始图像
        self.label_image.setFixedSize(1366, 768)  # 定义label的尺寸大小
        self.label_image.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)  # 对label进行边框设置
        self.label_image.setFrameShadow(QtWidgets.QFrame.Raised)  # 同上
        self.label_image.setLineWidth(3)  # 设定边框的粗细
        self.label_image.setStyleSheet('background-color: rgb(0, 255, 0)')  # 设定边框的颜色
        self.label_image.setScaledContents(True)  # 对label进行设置
        self.h_box_label_image = QHBoxLayout()  # 创建水平布局
        self.h_box_label_image.addWidget(self.label_image)  # 将label添加到水平布局

        try:  # 使用try避免程式异常终止
            YOLO(self)  # 调用YOLO函数，加载相关配置文件
            
            self.real_ng_sum = 0  # 定义初始值
            self.real_ok_sum = 0  # 同上
            self.photo_ng_mark = 0  # 同上
            self.real_ng_mark = 0  # 同上
            self.real_no_alarm_mark = 0  #同上
            self.ng_image_site = 1

            history_data_today = pd.read_csv("./history_data/history_data_today.csv")  # 读取当天历史数据
            history_data_today_df = pd.DataFrame(history_data_today)  # 数据转换为dataframe格式
            history_data_today_list = []  # 创建列表
            history_data_today_list.append(history_data_today_df.iloc[0, 1])  # 将OK数量写入列表
            history_data_today_list.append(history_data_today_df.iloc[0, 2])  # 将NG数量写入列表
            history_data_today_list.append(history_data_today_df.iloc[0, 3])  # 将Total数量写入列表
            history_data_today_list.append(history_data_today_df.iloc[0, 4])  # 将良率写入列表
            
            self.ok_sum = history_data_today_list[0]  # 数据赋给相应变量作为初始值
            self.ng_sum = history_data_today_list[1]  # 数据赋给相应变量作为初始值
            self.total_sum = history_data_today_list[2]  # 数据赋给相应变量作为初始值
            self.yield_sum = history_data_today_list[3]  # 数据赋给相应变量作为初始值
            self.ng_sum = 0

            # Pin Definitions
            self.output_pin = 11  # 定义GPIO使用的针脚为11
            GPIO.setmode(GPIO.BOARD)  # 设置GPIO模式为BOARD
            # set pin as an output pin with optional initial state of HIGH
            GPIO.setup(self.output_pin, GPIO.OUT, initial=GPIO.LOW)  # 对GPIO端口进行设置

            self.timer_camera = QtCore.QTimer()  # 定义定时器，用于YOLO实时监测
            self.timer_camera.timeout.connect(self.real_image)  # 设定定时器link函数
            self.timer_camera.start(100)  # 设定定时器每隔100ms（0.1s）执行一次
            self.camera_state = 1  # 设定摄像头状态为工作状态，便于后续log记录
        except:  # try内程序报错时执行下面程序
            self.camera_state = 0  # 设定摄像头状态为非工作状态（摄像头启动异常），便于后续log记录

        self.label_ng = QLabel(self)  # 定义label
        self.label_ng.setText('<b>未佩戴口罩人员数量统计<b>')  # 设定label显示文本
        self.label_ng.setFont(QFont('SanSerif', 24))  # 设定label显示文本的字体
        self.label_ng.setStyleSheet('color: rgb(255, 0, 0)')  # 设定label显示文本的颜色
        self.label_ng.setAlignment(Qt.AlignCenter)  # 设定文本的对齐方式（左对齐）
        self.label_ng.setFixedSize(400, 50)  # 设定label的尺寸
        self.h_box_ng = QHBoxLayout()  # 创建水平布局
        self.h_box_ng.addWidget(self.label_ng)  # 将label添加到布局

        self.lcd_ng = QLabel(self)  # 定义label
        self.lcd_ng.setText(str(self.ng_sum))  # 设定label显示文本
        self.lcd_ng.setFont(QFont('SanSerif', 24))  # 设定label显示文本的字体
        self.lcd_ng.setStyleSheet('color: rgb(255, 0, 0)')  # 设定label显示文本的颜色
        self.lcd_ng.setFrameShape(QtWidgets.QFrame.Box)  # 设定label的边框
        self.lcd_ng.setFrameShadow(QtWidgets.QFrame.Raised)  # 设定label的边框
        self.lcd_ng.setLineWidth(1)  # 设定label边框的粗细
        self.lcd_ng.setAlignment(Qt.AlignCenter)  # 设定显示文本的对齐方式（中心对齐）
        self.lcd_ng.setFixedSize(150, 50)  # 设定label的尺寸
        self.h_box_ng_num = QHBoxLayout()  # 创建水平布局
        self.h_box_ng_num.addWidget(self.lcd_ng)  # 将label添加到布局

        self.h_box_ng_label_num = QVBoxLayout()  # 创建水平布局，将NG文本和NG数量组合为一个布局
        self.h_box_ng_label_num.addLayout(self.h_box_ng)  # 添加布局到布局
        self.h_box_ng_label_num.addLayout(self.h_box_ng_num)  # 添加布局到布局

        self.label_image_ng_1 = QLabel(self)  # 定义label
        self.label_image_ng_1.setPixmap(QPixmap('./sources_file/image_sample.png'))  # 定义label的内容为图片，并显示初始图像
        self.label_image_ng_1.setFixedSize(196, 96)  # 定义label的尺寸大小
        self.label_image_ng_1.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_image_ng_1.setFrameShape(QtWidgets.QFrame.Box)  # 对label进行边框设置
        self.label_image_ng_1.setFrameShadow(QtWidgets.QFrame.Raised)  # 同上
        self.label_image_ng_1.setLineWidth(1)  # 设定边框的粗细
        self.label_image_ng_1.setStyleSheet('background-color: rgb(255, 0, 0)')  # 设定边框的颜色
        self.label_image_ng_1.setScaledContents(True)  # 对label进行设置
        self.h_box_label_image_ng_1 = QHBoxLayout()  # 创建水平布局
        self.h_box_label_image_ng_1.addWidget(self.label_image_ng_1)  # 将label添加到水平布局

        self.label_image_ng_2 = QLabel(self)  # 定义label
        self.label_image_ng_2.setPixmap(QPixmap('./sources_file/image_sample.png'))  # 定义label的内容为图片，并显示初始图像
        self.label_image_ng_2.setFixedSize(196, 96)  # 定义label的尺寸大小
        self.label_image_ng_2.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_image_ng_2.setFrameShape(QtWidgets.QFrame.Box)  # 对label进行边框设置
        self.label_image_ng_2.setFrameShadow(QtWidgets.QFrame.Raised)  # 同上
        self.label_image_ng_2.setLineWidth(1)  # 设定边框的粗细
        self.label_image_ng_2.setStyleSheet('background-color: rgb(255, 0, 0)')  # 设定边框的颜色
        self.label_image_ng_2.setScaledContents(True)  # 对label进行设置
        self.h_box_label_image_ng_2 = QHBoxLayout()  # 创建水平布局
        self.h_box_label_image_ng_2.addWidget(self.label_image_ng_2)  # 将label添加到水平布局

        self.label_image_ng_3 = QLabel(self)  # 定义label
        self.label_image_ng_3.setPixmap(QPixmap('./sources_file/image_sample.png'))  # 定义label的内容为图片，并显示初始图像
        self.label_image_ng_3.setFixedSize(196, 96)  # 定义label的尺寸大小
        self.label_image_ng_3.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_image_ng_3.setFrameShape(QtWidgets.QFrame.Box)  # 对label进行边框设置
        self.label_image_ng_3.setFrameShadow(QtWidgets.QFrame.Raised)  # 同上
        self.label_image_ng_3.setLineWidth(1)  # 设定边框的粗细
        self.label_image_ng_3.setStyleSheet('background-color: rgb(255, 0, 0)')  # 设定边框的颜色
        self.label_image_ng_3.setScaledContents(True)  # 对label进行设置
        self.h_box_label_image_ng_3 = QHBoxLayout()  # 创建水平布局
        self.h_box_label_image_ng_3.addWidget(self.label_image_ng_3)  # 将label添加到水平布局

        self.label_image_ng_4 = QLabel(self)  # 定义label
        self.label_image_ng_4.setPixmap(QPixmap('./sources_file/image_sample.png'))  # 定义label的内容为图片，并显示初始图像
        self.label_image_ng_4.setFixedSize(196, 96)  # 定义label的尺寸大小
        self.label_image_ng_4.setAlignment(Qt.AlignCenter)  # 定义label文字内容的对齐方式（中心对齐）
        self.label_image_ng_4.setFrameShape(QtWidgets.QFrame.Box)  # 对label进行边框设置
        self.label_image_ng_4.setFrameShadow(QtWidgets.QFrame.Raised)  # 同上
        self.label_image_ng_4.setLineWidth(1)  # 设定边框的粗细
        self.label_image_ng_4.setStyleSheet('background-color: rgb(255, 0, 0)')  # 设定边框的颜色
        self.label_image_ng_4.setScaledContents(True)  # 对label进行设置
        self.h_box_label_image_ng_4 = QHBoxLayout()  # 创建水平布局
        self.h_box_label_image_ng_4.addWidget(self.label_image_ng_4)  # 将label添加到水平布局

        self.label_camera = QLabel(self)  # 定义label
        self.label_camera.setPixmap(QPixmap('./sources_file/camera_black.png.png'))  # 设定label内容为图片，实现红绿灯显示
        self.label_camera.setFixedSize(300, 200)  # 设定label的尺寸
        self.label_camera.setScaledContents(True)  # 对label进行设置
        self.h_box_label_camera = QHBoxLayout()  # 创建水平布局
        self.h_box_label_camera.addWidget(self.label_camera)  # 将label添加到布局

        self.label_alarm = QLabel(self)  # 定义label
        self.label_alarm.setPixmap(QPixmap('./sources_file/green.png'))  # 设定label内容为图片，实现红绿灯显示
        self.alarm_state = 0  # 设定报警状态为0，绿灯状态，便于后续log档记录
        self.label_alarm.setFixedSize(100, 100)  # 设定label的尺寸
        self.label_alarm.setScaledContents(True)  # 对label进行设置
        self.h_box_label_alarm = QHBoxLayout()  # 创建水平布局
        self.h_box_label_alarm.addWidget(self.label_alarm)  # 将label添加到布局

        self.project_owner = QLabel(self)  # 定义label
        self.project_owner.setText("人体口罩佩戴实时检测方案提供者:艾聚达信息技术(苏州)有限公司(0512-62588800)")  # 设定label显示文本
        self.project_owner.setFont(QFont('SanSerif', 8))  # 设定label显示文本的字体
        self.project_owner.setStyleSheet('color: rgb(0, 255, 0)')  # 设定label显示文本的颜色
        self.project_owner.setFrameShape(QtWidgets.QFrame.Box)  # 设定label的边框
        self.project_owner.setFrameShadow(QtWidgets.QFrame.Raised)  # 设定label的边框
        self.project_owner.setLineWidth(1)  # 设定label边框的粗细
        self.project_owner.setAlignment(Qt.AlignCenter)  # 设定显示文本的对齐方式（中心对齐）
        self.project_owner.setFixedSize(440, 30)  # 设定label的尺寸
        self.h_box_project_owner = QHBoxLayout()  # 创建水平布局
        self.h_box_project_owner.addWidget(self.project_owner)  # 将label添加到布局

    def ui_addlayout_for_widget(self):  # 将局部布局进行组合形成新的布局

        self.h_box_owner_datetime = QHBoxLayout()  # 创建水平布局
        self.h_box_owner_datetime.addLayout(self.h_box_owner)  # 添加布局到布局
        self.h_box_owner_datetime.addStretch(1)  # 添加伸缩控件
        self.h_box_owner_datetime.addLayout(self.h_box_datetime)  # 添加布局到布局

        self.g_box_ng_image = QGridLayout()
        self.g_box_ng_image.addWidget(self.label_image_ng_1, 0, 0)
        self.g_box_ng_image.setSpacing(20)
        self.g_box_ng_image.addWidget(self.label_image_ng_2, 0, 1)
        self.g_box_ng_image.setSpacing(10)
        self.g_box_ng_image.addWidget(self.label_image_ng_3, 1, 0)
        self.g_box_ng_image.setSpacing(20)
        self.g_box_ng_image.addWidget(self.label_image_ng_4, 1, 1)

        self.v_box_data_alarm = QVBoxLayout()  # 添加布局到布局
        self.v_box_data_alarm.addLayout(self.h_box_label_camera)  # 添加布局到布局
        self.v_box_data_alarm.addStretch(1)  # 添加伸缩控件
        self.v_box_data_alarm.addLayout(self.h_box_label_alarm)  # 添加布局到布局
        self.v_box_data_alarm.addStretch(1)  # 添加伸缩控件
        self.v_box_data_alarm.addLayout(self.h_box_ng_label_num)  # 添加布局到布局
        self.v_box_data_alarm.addStretch(1)  # 添加伸缩控件
        self.v_box_data_alarm.addLayout(self.g_box_ng_image)  # 添加布局到布局
        self.v_box_data_alarm.addStretch(1)  # 添加伸缩控件
        self.v_box_data_alarm.addLayout(self.h_box_project_owner)  # 添加布局到布局

        self.h_box_image_data_alarm = QHBoxLayout()  # 创建水平布局
        self.h_box_image_data_alarm.addLayout(self.h_box_label_image)  # 添加布局到布局
        self.h_box_image_data_alarm.addLayout(self.v_box_data_alarm)  # 添加布局到布局

    def ui_addlayout_for_layout(self):  # 设置界面的整体布局

        self.v_box = QVBoxLayout()  # 创建垂直布局
        self.v_box.addStretch(1)  # 添加伸缩控件
        self.v_box.addLayout(self.h_box_title)  # 添加布局到布局
        self.v_box.addLayout(self.h_box_owner_datetime)  # 添加布局到布局
        self.v_box.addStretch(1)  # 添加伸缩控件
        self.v_box.addLayout(self.h_box_image_data_alarm)  # 添加布局到布局
        self.v_box.addStretch(1)  # 添加伸缩控件

        self.setLayout(self.v_box)  # 将整个界面的布局设定为创建好的垂直布局

    def date_time(self):  # 在此处实现实时时间的显示和部分界面数据的更新
        self.datetime_now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')  # 按照固定格式获取当前时间
        self.label_datetime.setText('<b>%s<b>' % self.datetime_now)  # 对时间label的文本进行重设

        if datetime.datetime.now().second % 2 == 0:
            self.label_camera.setPixmap(QPixmap('./sources_file/camera_black.png'))  # 设定label内容为图片，实现红绿灯显示
        if datetime.datetime.now().second % 2 == 1:
            self.label_camera.setPixmap(QPixmap('./sources_file/camera_white.png'))  # 设定label内容为图片，实现红绿灯显示

        self.history_data = "./history_data/history_data_today.csv"
        data_list = [0, self.ng_sum, 0, 0]
        data = pd.DataFrame(data=[data_list])
        data.to_csv(self.history_data, mode='w')


    def real_image(self):  # YOLO实时监测函数
        ret, self.frame_read = self.cap.read()  # 获取每帧图像
        self.frame_rgb = cv2.cvtColor(self.frame_read, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        self.frame_resized = cv2.resize(self.frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)  # 格式转换
        darknet.copy_image_from_bytes(self.darknet_image, self.frame_resized.tobytes())  # 调用darknet内部函数
        detections = darknet.detect_image(netMain, metaMain, self.darknet_image, thresh=0.75)  # 进行目标检测，thresh=0.25设定置信度
        self.image = cvDrawBoxes(self, detections, self.frame_resized)  # 调用识别框绘制函数，同时进行逻辑判断
        self.real_image = self.frame_rgb  # 图像赋给另一变量，实现界面显示
        self.real_image = QtGui.QImage(self.real_image[:], self.real_image.shape[1], self.real_image.shape[0], self.real_image.shape[1] * 3, QtGui.QImage.Format_RGB888)  # 对图像格式进行转换
        self.label_image.setPixmap(QPixmap(self.real_image))  # 更新label图像，实现视频流显示


if __name__ == '__main__':  # 函数从此处执行

    app = QApplication(sys.argv)  # 每一个PyQt5程序中都需要有一个QApplication对象，QApplication类包含在QTWidgets模块中，sys.argv是一个命令行参数列表；Python脚本可以从Shell中执行，比如双击*.py文件，通过参数来选择启动脚本的方式
    form = MainWindow()  # 类的实例化
    form.showFullScreen()  # 使用show()方法将窗口控件显示在屏幕上，全屏显示
    sys.exit(app.exec_())  # 进入该程序的主循环;使用sys.exit()方法的退出可以保证程序完整的结束，在这种情况下系统的环境变量会记录程序是如何退出的；如果程序运行成功，exec_()的返回值为0，否则为非0

    pass  # 不执行任何动作
