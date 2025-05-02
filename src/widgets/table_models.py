from PySide6.QtCore import Qt, QModelIndex, QAbstractTableModel
from PySide6.QtGui import QColor

class TableModel(QAbstractTableModel):
    """
    Base table model for displaying data in a table view.
    """
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = headers

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    def getData(self):
        return self._data

    def full_data(self, row, column):
        return self._data[row][column]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.EditRole:
            self._data[index.row()][index.column()] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

class CharacterTableModel(QAbstractTableModel):
    """
    Table model for displaying character data.
    """
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = headers

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    def full_data(self, row, column):
        return self._data[row][column]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]
        return None

class CustomTableModel(QAbstractTableModel):
    """
    Custom table model with selection functionality.
    """
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = headers
        self.selected_rows = set()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        elif role == Qt.BackgroundRole and index.row() in self.selected_rows:
            return QColor(Qt.lightGray)
        
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]
        elif orientation == Qt.Orientation.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        return None

    def toggle_selection(self, row):
        if row in self.selected_rows:
            self.selected_rows.remove(row)
        else:
            self.selected_rows.add(row)
        self.redraw(row)

    def redraw(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))

class CustomReferencesModel(QAbstractTableModel):
    """
    Custom table model for references with selection functionality.
    """
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = headers
        self.selected_rows = set()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        elif role == Qt.BackgroundRole and index.row() in self.selected_rows:
            return QColor(Qt.lightGray)
        
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]
        elif orientation == Qt.Orientation.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        return None

    def toggle_selection(self, row):
        if row in self.selected_rows:
            self.selected_rows.remove(row)
        else:
            self.selected_rows.add(row)
        self.redraw(row)

    def redraw(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))