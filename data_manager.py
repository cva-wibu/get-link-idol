import sqlite3
import os

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import cv2

from control import DATA_PATH, DATABASE_PATH


class DataManager(Dataset):
    def __init__(self, ):
        self.conn = sqlite3.connect(DATABASE_PATH)
        self.cur = self.conn.cursor()

        self.cur.execute('SELECT UNIQUE idol_name FROM picture')
        self.idol_list = self.cur.fetchall()

        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.idol_list)

    def __getitem__(self, idx):
        self.cur.execute('SELECT idol_name, pic_name FROM picture where ROWID == (?)', idx)
        idol_name, pic_name = self.cur.fetchall()
        path = os.path.join(DATA_PATH, pic_name)

        return self.__preprocessing(path), self.label_encoder.transform(idol_name)

    # TODO: handle multi label case
    def __len__(self):
        self.cur.execute('SELECT COUNT(pic_name) from picture')
        return self.cur.fetchone()

    # TODO: add preprocessing image
    def __preprocessing(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)
