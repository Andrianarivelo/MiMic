"""Results gallery: a reflowing grid of clickable spectrogram thumbnail cards.

Gives a fast, detailed visual overview of every file in the list. Clicking a
card opens that file in the interactive spectrogram view.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QLabel, QLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from . import theme


class FlowLayout(QLayout):
    """A layout that arranges children left-to-right and wraps to new rows."""

    def __init__(self, parent=None, margin=8, spacing=10):
        super().__init__(parent)
        self._items: list = []
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        line_height = 0
        right = rect.right() - m.right()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > right and line_height > 0:
                x = rect.x() + m.left()
                y = y + line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


class ThumbnailCard(QFrame):
    """One spectrogram thumbnail with a caption; emits its path when clicked."""

    clicked = Signal(str)

    def __init__(self, path: str, label: str):
        super().__init__()
        self.path = path
        self.setObjectName("ThumbCard")
        self.setFixedWidth(252)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(path)

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        self.image = QLabel()
        self.image.setFixedSize(238, 134)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setObjectName("ThumbImage")
        self.image.setText("…")
        v.addWidget(self.image)

        self.caption = QLabel(label)
        self.caption.setObjectName("ThumbCaption")
        self.caption.setWordWrap(False)
        self.caption.setTextInteractionFlags(Qt.NoTextInteraction)
        fm = self.caption.fontMetrics()
        self.caption.setText(fm.elidedText(label, Qt.ElideLeft, 234))
        v.addWidget(self.caption)

        self.subtitle = QLabel("")
        self.subtitle.setObjectName("ThumbSub")
        v.addWidget(self.subtitle)

    def set_image(self, pixmap: QPixmap) -> None:
        self.image.setText("")
        self.image.setPixmap(
            pixmap.scaled(238, 134, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_failed(self, msg: str) -> None:
        self.image.setText("⚠ failed")
        self.subtitle.setText(msg[:40])

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)


class GalleryView(QWidget):
    """Scrollable, reflowing grid of :class:`ThumbnailCard`."""

    card_clicked = Signal(str)

    def __init__(self):
        super().__init__()
        self._cards: dict[str, ThumbnailCard] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel("Build a gallery to see every file at a glance.")
        self.status.setObjectName("Hint")
        self.status.setContentsMargins(10, 6, 10, 2)
        outer.addWidget(self.status)

        self.container = QWidget()
        self.flow = FlowLayout(self.container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.container)
        outer.addWidget(scroll, stretch=1)

        self.setStyleSheet(
            f"""
            QFrame#ThumbCard {{
                background: {theme.BG_PANEL};
                border: 1px solid {theme.BORDER};
                border-radius: 8px;
            }}
            QFrame#ThumbCard:hover {{ border-color: {theme.ACCENT}; }}
            QLabel#ThumbImage {{
                background: {theme.BG_PLOT};
                border: 1px solid {theme.BORDER};
                border-radius: 4px;
                color: {theme.TEXT_FAINT};
            }}
            QLabel#ThumbCaption {{ color: {theme.TEXT}; font-size: 12px; }}
            QLabel#ThumbSub {{ color: {theme.TEXT_DIM}; font-size: 10px; }}
            """
        )

    def clear(self) -> None:
        while self.flow.count():
            item = self.flow.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards.clear()

    def add_card(self, path: str, label: str) -> ThumbnailCard:
        card = ThumbnailCard(path, label)
        card.clicked.connect(self.card_clicked)
        self.flow.addWidget(card)
        self._cards[path] = card
        return card

    def card(self, path: str) -> ThumbnailCard | None:
        return self._cards.get(path)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
