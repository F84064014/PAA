from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout,
    QGroupBox, QVBoxLayout,
    QHBoxLayout, QPushButton,
    QDoubleSpinBox
)
from PAA.backend import (
    Attributes
)
import numpy as np

class FilterPanel(QWidget):
    def __init__(self, attributes: Attributes, splits: list[str]):
        super().__init__()
        self.attr_cbs   :list[QCheckBox]        = list()
        self.prob_sbs   :list[QDoubleSpinBox]   = list()
        self.prob_pbs   :list[QPushButton]      = list()
        self.split_cbs  :list[QCheckBox]        = list()
        self.layout     :QVBoxLayout            = QVBoxLayout(self)

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setStyleSheet("background-color: silver; color: black")

        self.layout.addWidget(self.build_reset_widget())
        for group_box in self.build_attributes_cbs(attributes):
            self.layout.addWidget(group_box)
        self.layout.addWidget(self.build_split_cbs(splits))

    def build_attributes_cbs(self, attributes: Attributes) -> list[QGroupBox]:
        group_boxes = []
        for group_name, attr_list in attributes.group():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout()

            for i, (idx, attr) in enumerate(attr_list):
                container = QWidget()
                v_layout  = QVBoxLayout(container)
                v_layout.setContentsMargins(0, 0, 0, 0)
                v_layout.setSpacing(5)

                cb = QCheckBox(attr)
                cb.setCheckState(Qt.CheckState.Unchecked)
                self.attr_cbs.append(cb)

                row, col = divmod(i, 6)
                v_layout.addWidget(cb)
                v_layout.addWidget(self.build_prob_condition_widget())

                group_layout.addWidget(container, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    def build_split_cbs(self, splits: list[str]) -> QGroupBox:
        box = QGroupBox("Partition")
        layout = QHBoxLayout(box)
        for split in splits:
            cb = QCheckBox(split)
            cb.setChecked(True)
            self.split_cbs.append(cb)
            layout.addWidget(cb)
        return box

    def build_prob_condition_widget(self) -> QWidget:
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(5)

        # thre = QLineEdit()
        # thre.setText("1.0")
        thre = QDoubleSpinBox()
        thre.setRange(0.0, 1.0)
        thre.setSingleStep(0.1)
        thre.setFixedWidth(50)
        thre.setValue(1.0)
        # thre.setValidator(QDoubleValidator())
        self.prob_sbs.append(thre)

        sign = QPushButton('<')
        sign.setFixedWidth(30)
        sign.setCheckable(True)
        def update_sign():
            sign.setText(">" if sign.isChecked() else "<")
        sign.clicked.connect(update_sign)
        self.prob_pbs.append(sign)


        h_layout.addWidget(sign, alignment=Qt.AlignmentFlag.AlignLeft)
        h_layout.addWidget(thre, alignment=Qt.AlignmentFlag.AlignLeft)

        return container

    def build_reset_widget(self):
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        return reset_btn

    def reset(self) -> None:
        for pb, sb, cb in zip(self.prob_pbs, self.prob_sbs, self.attr_cbs):
            sb.setValue(1.0)
            pb.setText('<')
            pb.setChecked(False)
            cb.setChecked(False)

    def checkPredict(self, prob: np.array) -> bool:
        value = np.array([le.value() for le in self.prob_sbs], dtype=float)
        gt = np.array([pb.text()==">" for pb in self.prob_pbs], dtype=bool)
        lt = np.array([pb.text()=="<" for pb in self.prob_pbs], dtype=bool)

        gt_res = prob[gt] >= value[gt]
        lt_res = prob[lt] <= value[lt]

        return all(gt_res) and all(lt_res)

    @property
    def mask(self) -> np.ndarray:
        return np.array([cb.isChecked() for cb in self.attr_cbs], dtype=int)
    
    @property
    def split(self) -> set:
        return set([cb.text() for cb in self.split_cbs if cb.isChecked()])