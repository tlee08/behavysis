import sys

from PySide6.QtWidgets import QApplication

from behavysis.viewer.windows.main import MainWindow


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # window.open(
    #     # r"z:\PRJ-BowenLab\TimLee\resources\project_ma\0_configs\2_Round1.1_20220530_AGG-MOA_test3-M3_a2.json",
    #     "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/TimLee/resources/project_ma/0_configs/2_Round1.1_20220530_AGG-MOA_test3-M3_a2.json",
    # )
    # window.export_vid("example.mp4")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
