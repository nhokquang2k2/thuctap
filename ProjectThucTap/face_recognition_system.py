import os
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime, date

class FaceRecognitionSystem:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_attendance_path = os.path.join(base_dir, 'raw')
        self.dataset_path = os.path.join(base_dir, 'dataset')
        self.db_path = os.path.join(base_dir, 'data', 'data.db')
        self.encode_list_known = []
        self.classNames = []
        self.load_encodings()

    def save_raw_attendance(self, name):
        today = str(date.today())
        txt = "raw_" + today + ".txt"
        os.makedirs(self.raw_attendance_path, exist_ok=True)
        file = os.path.join(self.raw_attendance_path, txt)

        try:
            with open(file, "a+", newline="\n") as fp:
                my_data_list = fp.readlines()
                name_list = [line.split(",")[0] for line in my_data_list]
                if name not in name_list:
                    now = datetime.now()
                    dt_string = now.strftime("%H:%M:%S")
                    fp.writelines(name + ',' + dt_string + '\n')
        except Exception as ex:
            print(ex)

    def load_encodings(self):
        self.encode_list_known = []
        self.classNames = []

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.dataset_path, filename)
                encoding = np.load(filepath)
                self.encode_list_known.append(encoding)
                self.classNames.append(os.path.splitext(filename)[0])

        return self.encode_list_known, self.classNames

    def reload_encodings(self):
        self.load_encodings()
