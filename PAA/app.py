from PyQt6.QtCore import (
    Qt
)
from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QPushButton,
    QLabel, QHBoxLayout, QVBoxLayout,
    QApplication, QMenuBar
)
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
)
from .components import (
    ImageLabel, AttributeLabel, FilterPanel,
    ModelPanel
)
from .backend import (
    load_data
)
import os
import numpy as np
from datetime import datetime

class Annotator(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PAA Tool")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName("MainWindow")
        self.setStyleSheet("""
            QWidget#MainWindow {
                           background-color: #2b2b2b}
                           """)
        self.setFixedSize(1000, 800)

        # Loading dataset
        # self.dataset = load_data("data/RealWorld_0421.pth")
        self.dataset = load_data("data/RealWorld_0424.pth")
        self.dataset.append_split("train")
        self.dataset.append_split("val")
        self.dataset.append_split("ignore")

        self.imageLabel     :ImageLabel
        self.attributeLabel :AttributeLabel

        self.index_edit = QLineEdit(str(0))
        self.index_edit.setFixedWidth(50)
        self.index_edit.setStyleSheet("color: white; background-color: #3b3b3b;")
        self.index_edit.returnPressed.connect(self.jump_image)

        self.total_label = QLabel(f" / {len(self.dataset)}")
        self.total_label.setStyleSheet("color: white;")

        self.filter_btn = QPushButton("Filter")
        self.filter_btn.clicked.connect(self.toggle_filter_panel)
        self.filter_pannel = FilterPanel(
            self.dataset.attributes, self.dataset.split_names)

        self.model_btn = QPushButton("Model")
        self.model_btn.clicked.connect(self.toggle_model_panel)
        self.model_pannel = ModelPanel(
            "../PAR/exp/shufflenetv2_1.0_CBAM_finetune/shufflenetv2_1.0_CBAM_finetune.onnx", self.dataset.attributes)

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.index_edit)
        top_layout.addWidget(self.total_label)
        top_layout.addWidget(self.filter_btn)
        top_layout.addWidget(self.model_btn)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.setMenuBar(self.buildMenuBar())
        main_layout.addLayout(top_layout)
        main_layout.addLayout(self.buildImageAndAttribute())

        self.setLayout(main_layout)

        self._cur_index : int = 0
        self.load_image()

        self.pred : np.ndarray = None
        self.predict_dataset(fromfile=True)

    @property
    def cur_index(self):
        return self._cur_index

    @cur_index.setter
    def cur_index(self, value):
        if isinstance(value, str) and str.isdigit(value):
            value = int(value)
        self._cur_index = min(max(value, 0), len(self.dataset)-1)
        self.index_edit.setText(str(self._cur_index))

    def find_next(self, step=1):
        init_index = self._cur_index
        self._cur_index += step

        while self._cur_index >= 0 and self._cur_index < len(self.dataset):
            l = self.dataset.get_label(self._cur_index)
            s = self.dataset.get_split(self._cur_index)

            cp = (self.pred is None) or \
                 self.filter_pannel.checkPredict(self.pred[self._cur_index])
            cl = self.filter_pannel.checkLabel(l)
            cs = self.filter_pannel.checkSplit(s)

            if cl and cs and cp:
                self.cur_index = self._cur_index # Update UI
                return
            self._cur_index += step

        self.cur_index = init_index

    def jump_image(self):
        self.save_image()
        self.cur_index = self.index_edit.text()
        self.load_image()

    def load_image(self):
        self.imageLabel.loadImage(self.dataset.get_image(self.cur_index))
        self.attributeLabel.loadLabel(
            self.dataset.get_label(self.cur_index),
            self.dataset.get_split(self.cur_index),
        )

    def load_predict(self):
        if self.pred is None:
            self.model_pannel.updatePredict(
                self.dataset.get_image(self.cur_index),
                self.dataset.get_label(self.cur_index))
        else:
            self.model_pannel.updateProb(self.pred[self.cur_index])
            self.model_pannel.updateDiff(self.pred[self.cur_index],
                                         self.dataset.labels[self.cur_index])

    def save_image(self, force_split=None):
        label, split = self.attributeLabel.getLabel()
        if force_split: split = force_split
        self.dataset.set_label(self._cur_index, label)
        self.dataset.set_split(self._cur_index, split)

    def save_dataset(self):
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
        if self.save_csv_action.isChecked():
            self.dataset.save_csv(f"{timestamp}_temp.csv")
        elif self.save_pth_action.isChecked():
            self.dataset.save_pth(f"{timestamp}_temp.pth")
        else:
            print("[ERROR] No save format selected")

    def predict_dataset(self, fromfile=False):
        if fromfile:
            print("Start loading previous prediction")
            if not os.path.isfile("infer.npy"):
                print("[ERROR] infer.npy not found")
            else:
                self.pred = np.load("infer.npy")
                if len(self.pred) != len(self.dataset):
                    print("[ERROR] len(infer.npy)!=len(dataset), failed to load infer.npy")
                    self.pred = None
        else:
            print("Start predicting dataset [This might take times]")
            self.pred = self.model_pannel.model(self.dataset.image_paths)
            np.save("infer.npy", self.pred)
        print("predict_dtaset Done.")

    def toggle_filter_panel(self):
        if self.filter_pannel.isVisible():
            self.filter_pannel.hide()
        else:
            pos = self.filter_btn.mapToGlobal(
                self.filter_btn.rect().bottomLeft())
            self.filter_pannel.move(pos)
            self.filter_pannel.show()

    def toggle_model_panel(self):
        self.load_predict()
        if self.model_pannel.isVisible():
            self.model_pannel.hide()
        else:
            pos = self.model_btn.mapToGlobal(
                self.model_btn.rect().bottomLeft())
            self.model_pannel.move(pos)
            self.model_pannel.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.save_image()
            self.find_next(step=1)
            self.load_image()
            self.load_predict()
        if event.key() == Qt.Key.Key_D:
            self.save_image()
            self.find_next(step=-1)
            self.load_image()
            self.load_predict()
        if event.key() == Qt.Key.Key_S:
            self.save_image()
            self.save_dataset()
        if event.key() == Qt.Key.Key_Q:
            self.save_image()
            self.save_dataset()
            self.close()

        # Clean data
        if event.key() == Qt.Key.Key_I:
            self.save_image(force_split="ignore")
            self.find_next(step=1)
            self.load_image()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.save_dataset()
        event.accept()

    def buildImageAndAttribute(self) -> QHBoxLayout:

        self.imageLabel     = ImageLabel()
        self.attributeLabel = AttributeLabel(
            self.dataset.attributes,
            self.dataset.splits_name,
        )

        layout = QHBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.attributeLabel)

        return layout

    def buildMenuBar(self):
        menubar = QMenuBar()

        file_menu = menubar.addMenu("File")

        self.save_csv_action = QAction(".csv", self)
        self.save_csv_action.setCheckable(True)
        self.save_csv_action.setChecked(False)
        self.save_csv_action.triggered.connect(
            lambda x: self.save_pth_action.setChecked(False))
        file_menu.addAction(self.save_csv_action)

        self.save_pth_action = QAction(".pth", self)
        self.save_pth_action.setCheckable(True)
        self.save_pth_action.setChecked(True)
        self.save_pth_action.triggered.connect(
            lambda x: self.save_csv_action.setChecked(False))
        file_menu.addAction(self.save_pth_action)

        self.export_action = QAction("Export", self)
        self.export_action.triggered.connect(
            lambda x: self.dataset.export(
                f"data/ExportData_{datetime.now().strftime('%Y_%m%d')}",
                drop=["ignore"]))
        file_menu.addAction(self.export_action)

        model_menu = menubar.addMenu("Models")

        self.predict_dataset_action = QAction("predict")
        self.predict_dataset_action.triggered.connect(
            lambda x: self.predict_dataset(fromfile=False))
        model_menu.addAction(self.predict_dataset_action)

        self.load_predict_dataset_action = QAction("load predict")
        self.load_predict_dataset_action.triggered.connect(
            lambda x: self.predict_dataset(fromfile=True))
        model_menu.addAction(self.load_predict_dataset_action)

        self.unload_predict_dataset_action = QAction("unload predict")
        self.unload_predict_dataset_action.triggered.connect(
            lambda x: setattr(self, 'pred', None))
        model_menu.addAction(self.unload_predict_dataset_action)

        return menubar