"""One transactional disk file for sparse indexed working arrays; no raw images."""
from io import BytesIO
from pathlib import Path
import sqlite3
import numpy as np


class ArrayStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None

    def open(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.path)
            self.connection.execute('CREATE TABLE IF NOT EXISTS arrays (key TEXT PRIMARY KEY, value BLOB NOT NULL)')

    def write(self, key, **arrays):
        self.open()
        value = BytesIO()
        np.savez_compressed(value, **arrays)
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO arrays VALUES (?, ?)', (key, value.getvalue()))

    def contains(self, key):
        if self.connection is None and not self.path.exists():return False
        self.open()
        return self.connection.execute('SELECT 1 FROM arrays WHERE key=?', (key,)).fetchone() is not None

    def read(self, key):
        if not self.contains(key):raise KeyError(key)
        row = self.connection.execute('SELECT value FROM arrays WHERE key=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        with np.load(BytesIO(row[0]), allow_pickle=False) as value:
            return {name: value[name].copy() for name in value.files}

    def close(self):
        if self.connection is not None:self.connection.close();self.connection=None
