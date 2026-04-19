import sys
from .app import (
    Annotator, QApplication
)

if __name__=="__main__":
    app = QApplication(sys.argv)
    annotator = Annotator()
    annotator.show()
    sys.exit(app.exec())