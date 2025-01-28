# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settings_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QSizePolicy, QTabWidget,
    QWidget)

class Ui_SettingsWindow(object):
    def setupUi(self, SettingsWindow):
        if not SettingsWindow.objectName():
            SettingsWindow.setObjectName(u"SettingsWindow")
        SettingsWindow.resize(270, 222)
        self.general_tab = QWidget()
        self.general_tab.setObjectName(u"general_tab")
        self.formLayout = QFormLayout(self.general_tab)
        self.formLayout.setObjectName(u"formLayout")
        self.vid_header_lb = QLabel(self.general_tab)
        self.vid_header_lb.setObjectName(u"vid_header_lb")

        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.vid_header_lb)

        self.graph_header_lb = QLabel(self.general_tab)
        self.graph_header_lb.setObjectName(u"graph_header_lb")

        self.formLayout.setWidget(4, QFormLayout.SpanningRole, self.graph_header_lb)

        self.window_size_lb = QLabel(self.general_tab)
        self.window_size_lb.setObjectName(u"window_size_lb")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.window_size_lb)

        self.window_size_le = QLineEdit(self.general_tab)
        self.window_size_le.setObjectName(u"window_size_le")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.window_size_le)

        self.confirm_dbtn = QDialogButtonBox(self.general_tab)
        self.confirm_dbtn.setObjectName(u"confirm_dbtn")
        self.confirm_dbtn.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.confirm_dbtn)

        self.focus_size_lb = QLabel(self.general_tab)
        self.focus_size_lb.setObjectName(u"focus_size_lb")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.focus_size_lb)

        self.vid_speed_lb = QLabel(self.general_tab)
        self.vid_speed_lb.setObjectName(u"vid_speed_lb")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.vid_speed_lb)

        self.vid_speed_le = QLineEdit(self.general_tab)
        self.vid_speed_le.setObjectName(u"vid_speed_le")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.vid_speed_le)

        self.focus_size_le = QLineEdit(self.general_tab)
        self.focus_size_le.setObjectName(u"focus_size_le")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.focus_size_le)

        self.vid_width_le = QLineEdit(self.general_tab)
        self.vid_width_le.setObjectName(u"vid_width_le")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.vid_width_le)

        self.vid_width_lb = QLabel(self.general_tab)
        self.vid_width_lb.setObjectName(u"vid_width_lb")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.vid_width_lb)

        SettingsWindow.addTab(self.general_tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        SettingsWindow.addTab(self.tab_2, "")
#if QT_CONFIG(shortcut)
        self.window_size_lb.setBuddy(self.window_size_le)
        self.vid_width_lb.setBuddy(self.vid_width_le)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.vid_speed_le, self.vid_width_le)
        QWidget.setTabOrder(self.vid_width_le, self.window_size_le)
        QWidget.setTabOrder(self.window_size_le, self.focus_size_le)

        self.retranslateUi(SettingsWindow)

        QMetaObject.connectSlotsByName(SettingsWindow)
    # setupUi

    def retranslateUi(self, SettingsWindow):
        self.vid_header_lb.setText(QCoreApplication.translate("SettingsWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-weight:700;\">Video Player</span></p></body></html>", None))
        self.graph_header_lb.setText(QCoreApplication.translate("SettingsWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-weight:700;\">Graph viewer</span></p></body></html>", None))
        self.window_size_lb.setText(QCoreApplication.translate("SettingsWindow", u"Viewed secs", None))
        self.focus_size_lb.setText(QCoreApplication.translate("SettingsWindow", u"Focus padding secs", None))
        self.vid_speed_lb.setText(QCoreApplication.translate("SettingsWindow", u"x Speed", None))
        self.vid_width_lb.setText(QCoreApplication.translate("SettingsWindow", u"Width", None))
        SettingsWindow.setTabText(SettingsWindow.indexOf(self.general_tab), QCoreApplication.translate("SettingsWindow", u"General", None))
        SettingsWindow.setTabText(SettingsWindow.indexOf(self.tab_2), QCoreApplication.translate("SettingsWindow", u"Tab 2", None))
        pass
    # retranslateUi

