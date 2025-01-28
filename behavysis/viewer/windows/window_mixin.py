class WindowMixin:
    @staticmethod
    def toggle_window(widget):
        if widget.isVisible():
            widget.hide()
        else:
            widget.show()
