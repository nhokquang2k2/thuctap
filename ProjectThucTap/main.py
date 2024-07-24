import os
import sys
import cv2
import numpy as np
import face_recognition
import tensorflow as tf
import imutils
from datetime import datetime, timedelta
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from face_recognition_system import FaceRecognitionSystem
from login import show_login_popup, validate_login
from register import show_register_popup, register_user, capture_image, close_camera, show_captured_image_on_register_form
from database import init_db, get_users, get_unapproved_users, approve_user as db_approve_user, save_attendance, get_attendance_records, authenticate_user, get_user_image, delete_user as db_delete_user
from user_dashboard import show_user_dashboard
from admin_dashboard import show_admin_dashboard, show_approve_users_popup, show_attendance_records, show_manage_users_popup

class FaceRecognitionApp(App):
    def build(self):
        self.title = "Face Recognition and Liveness Detection"
        
        self.main_layout = BoxLayout(orientation='vertical')

        self.top_bar = BoxLayout(size_hint_y=None, height='48dp')
        self.login_button = Button(text='Login', size_hint_x=None, width='100dp', on_press=self.show_login)
        self.register_button = Button(text='Register', size_hint_x=None, width='100dp', on_press=self.show_register)
        self.dashboard_button = Button(text='Dashboard', size_hint_x=None, width='150dp', on_press=self.show_dashboard)
        self.dashboard_button.disabled = True

        self.top_bar.add_widget(BoxLayout()) 
        self.top_bar.add_widget(self.login_button)
        self.top_bar.add_widget(self.register_button)
        self.top_bar.add_widget(self.dashboard_button)

        self.main_layout.add_widget(self.top_bar)

        self.image = Image(size_hint=(1, None), height=480) 
        self.main_layout.add_widget(self.image)

        self.result_label = Label(text="Đang chờ...")
        self.main_layout.add_widget(self.result_label)

        self.bottom_bar = BoxLayout(size_hint_y=None, height='48dp')
        self.toggle_camera_button = Button(text='Start Camera', on_press=self.toggle_camera)
        self.confirm_button = Button(text='Confirm', on_press=self.confirm_attendance, disabled=True)
        self.bottom_bar.add_widget(self.toggle_camera_button)
        self.bottom_bar.add_widget(self.confirm_button)

        self.main_layout.add_widget(self.bottom_bar)

        self.cap = None
        self.detector_net = None
        self.liveness_model = None
        self.face_recognition_system = FaceRecognitionSystem()
        self.current_name = ""
        self.current_label = ""
        self.start_time = None
        self.logged_in_role = None
        self.username = None
        self.profile_image = None
        self.captured_image = None
        self.register_popup = None
        self.camera_popup = None
        self.captured_image_popup = None
        self.login_popup = None
        self.recognition_start_time = None

        init_db()

        return self.main_layout

    def show_login(self, instance):
        show_login_popup(self)

    def validate_login(self, instance):
        username = self.login_username_input.text
        password = self.login_password_input.text
        user = authenticate_user(username, password)
        if user:
            self.username = username
            self.logged_in_role = user[3] 
            self.dashboard_button.disabled = False
            self.login_popup.dismiss()
            self.show_dashboard()
        else:
            self.show_notification("Sai tên người dùng hoặc mật khẩu hoặc tài khoản chưa được chấp nhận từ admin")

    def show_register(self, instance):
        show_register_popup(self)

    def close_register_popup(self, instance):
        if self.register_popup:
            self.register_popup.dismiss()

    def register_user(self, instance):
        register_user(self, instance)

    def show_dashboard(self, instance=None):
        if self.logged_in_role == 'admin':
            show_admin_dashboard(self)
        elif self.logged_in_role == 'user':
            show_user_dashboard(self)
        else:
            self.main_layout.clear_widgets()
            self.main_layout.add_widget(self.top_bar)

    def show_approve_users_popup(self, instance=None):
        if self.logged_in_role == 'admin':
            show_approve_users_popup(self)
        else:
            self.result_label.text = "Bị từ chối, chỉ dành cho admin"

    def approve_user(self, user):
        db_approve_user(user)
        self.face_recognition_system.reload_encodings()
        self.show_approve_users_popup()

    def delete_user(self, username):
        db_delete_user(username)
        self.face_recognition_system.reload_encodings()
        self.show_notification(f"User {username} xóa thành công.")
        self.show_manage_users_popup()

    def show_attendance_records(self, instance):
        if self.logged_in_role == 'admin':
            show_attendance_records(self)
        else:
            self.result_label.text = "Bị từ chối, chỉ dành cho admin"

    def show_manage_users_popup(self, instance=None):
        if self.logged_in_role == 'admin':
            show_manage_users_popup(self)
        else:
            self.result_label.text = "Bị từ chối, chỉ dành cho admin"

    def logout(self, instance=None):
        self.logged_in_role = None
        self.username = None
        self.show_main_screen()

    def show_main_screen(self, instance=None):
        self.main_layout.clear_widgets()
        self.main_layout.add_widget(self.top_bar)
        self.main_layout.add_widget(self.image)
        self.main_layout.add_widget(self.result_label)
        self.main_layout.add_widget(self.bottom_bar)
        self.dashboard_button.disabled = True
        self.result_label.text = "Đăng xuất thành công."

    def toggle_camera(self, instance):
        if self.cap is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        if self.detector_net is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            proto_path = os.path.join(base_dir, 'face_recognition_and_liveness/face_liveness_detection/face_detector/deploy.prototxt')
            caffe_model_path = os.path.join(base_dir, 'face_recognition_and_liveness/face_liveness_detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
            self.detector_net = cv2.dnn.readNetFromCaffe(proto_path, caffe_model_path)

        if self.liveness_model is None:
            model_path = os.path.join(base_dir, 'face_recognition_and_liveness/face_liveness_detection/modelll.h5')
            self.liveness_model = tf.keras.models.load_model(model_path)

        self.result_label.text = "Camera started"
        self.toggle_camera_button.text = 'Stop Camera'
        self.reset_recognition_state()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            Clock.unschedule(self.update_frame)
            self.result_label.text = "Camera stopped"
            self.toggle_camera_button.text = 'Start Camera'
            self.confirm_button.disabled = True
            self.image.texture = None
            self.reset_recognition_state()

    def reset_recognition_state(self):
        self.current_name = ""
        self.current_label = ""
        self.recognition_start_time = None
        self.confirm_button.disabled = True

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.detector_net.setInput(blob)
        detections = self.detector_net.forward()

        detected_real = False

        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                startX = max(0, startX-20)
                startY = max(0, startY-20)
                endX = min(w, endX+20)
                endY = min(h, endY+20)

                face = frame[startY:endY, startX:endX]
                face_to_recog = face

                try:
                    face = cv2.resize(face, (32, 32))
                except:
                    continue

                rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                name = 'Unknown'

                if len(encodings) > 0:
                    encoding = encodings[0]
                    matches = face_recognition.compare_faces(self.face_recognition_system.encode_list_known, encoding)
                    face_dis = face_recognition.face_distance(self.face_recognition_system.encode_list_known, encoding)

                    if len(face_dis) > 0:
                        best_match_index = np.argmin(face_dis)
                        if matches[best_match_index] and face_dis[best_match_index] < 0.4:
                            name = self.face_recognition_system.classNames[best_match_index]
                            self.face_recognition_system.save_raw_attendance(name)
                        else:
                            name = 'Unknown'

                face = face.astype('float') / 255.0
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = self.liveness_model.predict(face)[0]
                label_name = 'real' if np.argmax(preds) == 1 else 'fake'

                if label_name == 'real':
                    detected_real = True
                    if self.current_name != name:
                        self.current_name = name
                        self.recognition_start_time = datetime.now()
                    elif self.recognition_start_time is not None and datetime.now() - self.recognition_start_time >= timedelta(seconds=5):
                        if name != 'Unknown':
                            self.confirm_button.disabled = False
                            print(f"Confirm button enabled for {name}")
                        else:
                            self.confirm_button.disabled = True
                else:
                    self.current_name = ""
                    self.recognition_start_time = None
                    self.confirm_button.disabled = True

                label = f'{label_name}: {preds[np.argmax(preds)]:.4f}'
                self.result_label.text = f'{name}, {label_name}'

                if label_name == 'fake':
                    cv2.putText(frame, "Đừng cố giả mạo!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)

        if not detected_real:
            self.recognition_start_time = None
            self.confirm_button.disabled = True

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def update_camera_frame(self, dt):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = imutils.resize(frame, width=800)
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.capture_image_widget.texture = texture

    def confirm_attendance(self, instance):
        if self.current_name != "" and self.result_label.text.split(", ")[-1] == "real":
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            save_attendance(self.current_name, current_time)
            self.show_notification(f"Chấm công cho {self.current_name} vào lúc {current_time}")
            self.current_name = ""
            self.current_label = ""
            self.start_time = None
            self.confirm_button.disabled = True
        else:
            self.show_notification("Không nhận diện được")

    def show_notification(self, message):
        popup = ModalView(size_hint=(0.5, 0.3))
        box = BoxLayout(orientation='vertical', padding=10)
        label = Label(text=message)
        box.add_widget(label)
        btn = Button(text='Close', size_hint_y=0.2)
        btn.bind(on_press=popup.dismiss)
        box.add_widget(btn)
        popup.add_widget(box)
        popup.open()

    def get_user_profile_image(self, username):
        image_data = get_user_image(username)
        if image_data and image_data[0]:
            return image_data[0]
        return None

    def convert_image_to_texture(self, image_data):
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def get_unapproved_users(self):
        return get_unapproved_users()

    def get_users(self):
        return get_users()

    def get_attendance_records(self):
        return get_attendance_records()

    def on_stop(self):
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    init_db()
    FaceRecognitionApp().run()
