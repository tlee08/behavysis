# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListView, QMainWindow,
    QMenu, QMenuBar, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QStatusBar, QWidget)

from behavysis.viewer.widgets.cv_view import CvView
from behavysis.viewer.widgets.graph_view import GraphView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(683, 496)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        palette = QPalette()
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(174, 214, 241, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        brush2 = QBrush(QColor(255, 255, 255, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Light, brush2)
        brush3 = QBrush(QColor(214, 234, 248, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Midlight, brush3)
        brush4 = QBrush(QColor(87, 107, 120, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush4)
        brush5 = QBrush(QColor(116, 143, 161, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush2)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette.setBrush(QPalette.Active, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Active, QPalette.AlternateBase, brush3)
        brush6 = QBrush(QColor(255, 255, 220, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Active, QPalette.ToolTipText, brush)
        brush7 = QBrush(QColor(0, 0, 0, 127))
        brush7.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush7)
#endif
        brush8 = QBrush(QColor(0, 0, 0, 216))
        brush8.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush8)
        brush9 = QBrush(QColor(236, 236, 236, 255))
        brush9.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush9)
        palette.setBrush(QPalette.Inactive, QPalette.Light, brush2)
        brush10 = QBrush(QColor(245, 245, 245, 255))
        brush10.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Midlight, brush10)
        brush11 = QBrush(QColor(191, 191, 191, 255))
        brush11.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush11)
        brush12 = QBrush(QColor(169, 169, 169, 255))
        brush12.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Mid, brush12)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush8)
        palette.setBrush(QPalette.Inactive, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush9)
        palette.setBrush(QPalette.Inactive, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush10)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush)
        brush13 = QBrush(QColor(0, 0, 0, 63))
        brush13.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush13)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Light, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush10)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush13)
#endif
        MainWindow.setPalette(palette)
        self.action_open = QAction(MainWindow)
        self.action_open.setObjectName(u"action_open")
        self.action_settings = QAction(MainWindow)
        self.action_settings.setObjectName(u"action_settings")
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        self.action_quit = QAction(MainWindow)
        self.action_quit.setObjectName(u"action_quit")
        self.action_save = QAction(MainWindow)
        self.action_save.setObjectName(u"action_save")
        self.action_save_as_bouts = QAction(MainWindow)
        self.action_save_as_bouts.setObjectName(u"action_save_as_bouts")
        self.action_save_as_frames = QAction(MainWindow)
        self.action_save_as_frames.setObjectName(u"action_save_as_frames")
        self.action_export_video = QAction(MainWindow)
        self.action_export_video.setObjectName(u"action_export_video")
        self.action_help = QAction(MainWindow)
        self.action_help.setObjectName(u"action_help")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.vid_viewer = CvView(self.centralwidget)
        self.vid_viewer.setObjectName(u"vid_viewer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.vid_viewer.sizePolicy().hasHeightForWidth())
        self.vid_viewer.setSizePolicy(sizePolicy1)
        self.vid_viewer.setStyleSheet(u"background-color: rgb(4, 172, 255);")

        self.gridLayout.addWidget(self.vid_viewer, 2, 0, 1, 3)

        self.start_stop_btn = QPushButton(self.centralwidget)
        self.start_stop_btn.setObjectName(u"start_stop_btn")
        self.start_stop_btn.setCheckable(True)

        self.gridLayout.addWidget(self.start_stop_btn, 6, 0, 1, 1)

        self.bouts_view = QListView(self.centralwidget)
        self.bouts_view.setObjectName(u"bouts_view")

        self.gridLayout.addWidget(self.bouts_view, 2, 3, 1, 2)

        self.bout_replay_btn = QPushButton(self.centralwidget)
        self.bout_replay_btn.setObjectName(u"bout_replay_btn")

        self.gridLayout.addWidget(self.bout_replay_btn, 6, 3, 1, 1)

        self.vid_back_btn = QPushButton(self.centralwidget)
        self.vid_back_btn.setObjectName(u"vid_back_btn")

        self.gridLayout.addWidget(self.vid_back_btn, 6, 1, 1, 1)

        self.vid_fwd_btn = QPushButton(self.centralwidget)
        self.vid_fwd_btn.setObjectName(u"vid_fwd_btn")

        self.gridLayout.addWidget(self.vid_fwd_btn, 6, 2, 1, 1)

        self.bout_inspect_widget = QWidget(self.centralwidget)
        self.bout_inspect_widget.setObjectName(u"bout_inspect_widget")
        self.bout_inspect_widget.setEnabled(False)
        self.bout_inspect_grid = QGridLayout(self.bout_inspect_widget)
        self.bout_inspect_grid.setObjectName(u"bout_inspect_grid")
        self.bout_inspect_header = QLabel(self.bout_inspect_widget)
        self.bout_inspect_header.setObjectName(u"bout_inspect_header")

        self.bout_inspect_grid.addWidget(self.bout_inspect_header, 0, 0, 1, 2)

        self.bout_inspect_view = QListView(self.bout_inspect_widget)
        self.bout_inspect_view.setObjectName(u"bout_inspect_view")
        self.bout_inspect_view.setEnabled(False)

        self.bout_inspect_grid.addWidget(self.bout_inspect_view, 2, 0, 1, 2)

        self.behav_rbtns_group = QGroupBox(self.bout_inspect_widget)
        self.behav_rbtns_group.setObjectName(u"behav_rbtns_group")
        self.horizontalLayout = QHBoxLayout(self.behav_rbtns_group)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.is_behav_rbtn = QRadioButton(self.behav_rbtns_group)
        self.is_behav_rbtn.setObjectName(u"is_behav_rbtn")

        self.horizontalLayout.addWidget(self.is_behav_rbtn)

        self.not_behav_rbtn = QRadioButton(self.behav_rbtns_group)
        self.not_behav_rbtn.setObjectName(u"not_behav_rbtn")

        self.horizontalLayout.addWidget(self.not_behav_rbtn)

        self.select_behav_rbtn = QRadioButton(self.behav_rbtns_group)
        self.select_behav_rbtn.setObjectName(u"select_behav_rbtn")

        self.horizontalLayout.addWidget(self.select_behav_rbtn)


        self.bout_inspect_grid.addWidget(self.behav_rbtns_group, 1, 0, 1, 2)


        self.gridLayout.addWidget(self.bout_inspect_widget, 4, 3, 1, 2)

        self.bout_focus_btn = QPushButton(self.centralwidget)
        self.bout_focus_btn.setObjectName(u"bout_focus_btn")
        self.bout_focus_btn.setCheckable(True)

        self.gridLayout.addWidget(self.bout_focus_btn, 6, 4, 1, 1)

        self.graph_viewer = GraphView(self.centralwidget)
        self.graph_viewer.setObjectName(u"graph_viewer")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.graph_viewer.sizePolicy().hasHeightForWidth())
        self.graph_viewer.setSizePolicy(sizePolicy2)
        self.graph_viewer.setMinimumSize(QSize(10, 0))
        self.graph_viewer.setStyleSheet(u"background-color: rgb(255, 80, 68);")

        self.gridLayout.addWidget(self.graph_viewer, 4, 0, 1, 3)

        self.slider = QSlider(self.centralwidget)
        self.slider.setObjectName(u"slider")
        sizePolicy1.setHeightForWidth(self.slider.sizePolicy().hasHeightForWidth())
        self.slider.setSizePolicy(sizePolicy1)
        self.slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout.addWidget(self.slider, 5, 0, 1, 3)

        self.annot_keypts_cbx = QCheckBox(self.centralwidget)
        self.annot_keypts_cbx.setObjectName(u"annot_keypts_cbx")

        self.gridLayout.addWidget(self.annot_keypts_cbx, 5, 3, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 683, 23))
        self.menubar.setNativeMenuBar(False)
        self.menu_file = QMenu(self.menubar)
        self.menu_file.setObjectName(u"menu_file")
        self.menu_save = QMenu(self.menu_file)
        self.menu_save.setObjectName(u"menu_save")
        self.menu_export = QMenu(self.menu_file)
        self.menu_export.setObjectName(u"menu_export")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu_file.menuAction())
        self.menu_file.addAction(self.action_settings)
        self.menu_file.addAction(self.action_open)
        self.menu_file.addAction(self.menu_save.menuAction())
        self.menu_file.addAction(self.menu_export.menuAction())
        self.menu_file.addAction(self.action_help)
        self.menu_file.addAction(self.action_quit)
        self.menu_save.addAction(self.action_save)
        self.menu_save.addAction(self.action_save_as_frames)
        self.menu_save.addAction(self.action_save_as_bouts)
        self.menu_export.addAction(self.action_export_video)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.action_open.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.action_settings.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.action_quit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
        self.action_save.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.action_save_as_bouts.setText(QCoreApplication.translate("MainWindow", u"Save as bouts", None))
        self.action_save_as_frames.setText(QCoreApplication.translate("MainWindow", u"Save as", None))
        self.action_export_video.setText(QCoreApplication.translate("MainWindow", u"Export video", None))
        self.action_help.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.vid_viewer.setText("")
        self.start_stop_btn.setText(QCoreApplication.translate("MainWindow", u"start/stop", None))
        self.bout_replay_btn.setText(QCoreApplication.translate("MainWindow", u"replay bout", None))
        self.vid_back_btn.setText(QCoreApplication.translate("MainWindow", u"<5s", None))
        self.vid_fwd_btn.setText(QCoreApplication.translate("MainWindow", u">5s", None))
        self.bout_inspect_header.setText(QCoreApplication.translate("MainWindow", u"Behaviour Inspector", None))
        self.behav_rbtns_group.setTitle("")
        self.is_behav_rbtn.setText(QCoreApplication.translate("MainWindow", u"IS behav", None))
        self.not_behav_rbtn.setText(QCoreApplication.translate("MainWindow", u"NOT behav", None))
        self.select_behav_rbtn.setText(QCoreApplication.translate("MainWindow", u"Not sure..", None))
        self.bout_focus_btn.setText(QCoreApplication.translate("MainWindow", u"focus on bout", None))
        self.annot_keypts_cbx.setText(QCoreApplication.translate("MainWindow", u"Show Keypoints", None))
        self.menu_file.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menu_save.setTitle(QCoreApplication.translate("MainWindow", u"Save", None))
        self.menu_export.setTitle(QCoreApplication.translate("MainWindow", u"Export", None))
    # retranslateUi

