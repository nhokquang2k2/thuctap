from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
import cv2
import face_recognition
import numpy as np
import os
import tensorflow as tf
from kivy.clock import Clock
import imutils
from kivy.graphics.texture import Texture
from datetime import datetime, timedelta
from src.database import get_users, register_user as db_register_user, approve_user as db_approve_user, user_exists, authenticate_user
from face_recognition_system import FaceRecognitionSystem

def show_register_popup(app_instance):
    content = BoxLayout(orientation='horizontal', padding=10, spacing=10)
    
    form_layout = BoxLayout(orientation='vertical', spacing=10)
    app_instance.register_username_input = TextInput(hint_text='Username', multiline=False, size_hint_y=None, height=40)
    app_instance.register_password_input = TextInput(hint_text='Password', multiline=False, password=True, size_hint_y=None, height=40)
    app_instance.register_role_input = TextInput(hint_text='Role', multiline=False, size_hint_y=None, height=40)
    open_camera_btn = Button(text='Open Camera', on_press=lambda x: open_camera(app_instance), size_hint_y=None, height=40)
    app_instance.register_btn = Button(text='Register', on_press=lambda x: register_user(app_instance), size_hint_y=None, height=40, disabled=True)
    back_btn = Button(text='Back', on_press=app_instance.close_register_popup, size_hint_y=None, height=40)
    
    form_layout.add_widget(app_instance.register_username_input)
    form_layout.add_widget(app_instance.register_password_input)
    form_layout.add_widget(app_instance.register_role_input)
    form_layout.add_widget(open_camera_btn)
    form_layout.add_widget(app_instance.register_btn)
    form_layout.add_widget(back_btn)
    
    image_layout = BoxLayout(orientation='vertical', spacing=10)
    app_instance.captured_image_widget = Image(size_hint_y=None, height=200)
    image_layout.add_widget(app_instance.captured_image_widget)
    
    content.add_widget(form_layout)
    content.add_widget(image_layout)

    app_instance.register_popup = Popup(title='Register', content=content, size_hint=(0.8, 0.8))
    app_instance.register_popup.open()
    
    app_instance.register_username_input.bind(text=lambda instance, value: check_register_conditions(app_instance))
    app_instance.register_password_input.bind(text=lambda instance, value: check_register_conditions(app_instance))
    app_instance.register_role_input.bind(text=lambda instance, value: check_register_conditions(app_instance))

def open_camera(app_instance):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    app_instance.capture_image_widget = Image()
    app_instance.capture_btn = Button(text='Capture', on_press=lambda x: capture_image(app_instance), size_hint_y=None, height=40, disabled=True)
    back_btn = Button(text='Back', on_press=lambda x: close_camera(app_instance), size_hint_y=None, height=40)
    
    content.add_widget(app_instance.capture_image_widget)
    content.add_widget(app_instance.capture_btn)
    content.add_widget(back_btn)

    app_instance.camera_popup = Popup(title='Capture Image', content=content, size_hint=(0.8, 0.8))
    app_instance.camera_popup.open()
    
    app_instance.cap = cv2.VideoCapture(0)
    
    if app_instance.detector_net is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proto_path = os.path.join(base_dir, 'face_recognition_and_liveness', 'face_liveness_detection', 'face_detector', 'deploy.prototxt')
        caffe_model_path = os.path.join(base_dir, 'face_recognition_and_liveness', 'face_liveness_detection', 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
        app_instance.detector_net = cv2.dnn.readNetFromCaffe(proto_path, caffe_model_path)

    if app_instance.liveness_model is None:
        model_path = os.path.join(base_dir, 'face_recognition_and_liveness', 'face_liveness_detection', 'modelll.h5')
        app_instance.liveness_model = tf.keras.models.load_model(model_path)

    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.reload_encodings()
    app_instance.encode_list_known = face_recognition_system.encode_list_known
    app_instance.classNames = face_recognition_system.classNames

    app_instance.recognition_start_time = None
    app_instance.is_unknown_for_5_seconds = False
    Clock.schedule_interval(lambda dt: update_camera_frame(app_instance), 1.0 / 30.0)

def close_camera(app_instance):
    if app_instance.cap is not None:
        app_instance.cap.release()
        app_instance.cap = None
        Clock.unschedule(lambda dt: update_camera_frame(app_instance))
    app_instance.camera_popup.dismiss()

def capture_image(app_instance):
    if app_instance.cap is not None:
        ret, frame = app_instance.cap.read()
        if ret:
            app_instance.captured_image = frame
            app_instance.capture_btn.disabled = False
            show_captured_image_on_register_form(app_instance)
            check_register_conditions(app_instance)

def show_captured_image_on_register_form(app_instance):
    buf = cv2.flip(app_instance.captured_image, 0).tobytes()
    texture = Texture.create(size=(app_instance.captured_image.shape[1], app_instance.captured_image.shape[0]), colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    app_instance.captured_image_widget.texture = texture
    close_camera(app_instance)

def check_register_conditions(app_instance):
    if (app_instance.register_username_input.text and
        app_instance.register_password_input.text and
        app_instance.register_role_input.text and
        app_instance.captured_image is not None):
        app_instance.register_btn.disabled = False
    else:
        app_instance.register_btn.disabled = True

def register_user(app_instance, instance=None):
    username = app_instance.register_username_input.text
    password = app_instance.register_password_input.text
    role = app_instance.register_role_input.text

    if app_instance.captured_image is None:
        app_instance.show_notification("Không có ảnh ở đây, không thể tạo tài khoản")
        return

    if user_exists(username):
        app_instance.show_notification("Tên người dùng đã tồn tại, vui lòng sử dụng tên người dùng khác")
        return

    image = app_instance.captured_image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) == 0:
        app_instance.show_notification("Không nhận diện được gương mặt trong ảnh")
        return

    encoding = encodings[0]
    dataset_path = os.path.join('dataset', f'{username}.npy')
    os.makedirs('dataset', exist_ok=True)
    np.save(dataset_path, encoding)
    
    image_path = os.path.join('profile_images', f'{username}.png')
    os.makedirs('profile_images', exist_ok=True)
    cv2.imwrite(image_path, image)
    
    db_register_user(username, password, role, image_path)
    app_instance.register_popup.dismiss()
    app_instance.show_notification(f"User {username} đã đăng kí thành công, chờ xét duyệt")
    app_instance.captured_image = None

def show_approve_users_popup(app_instance, instance=None):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    scroll_view = ScrollView()
    users_layout = BoxLayout(orientation='vertical', size_hint_y=None)
    users_layout.bind(minimum_height=users_layout.setter('height'))

    for user in get_users():
        if user[2] == 0:
            user_layout = BoxLayout(size_hint_y=None, height=40)
            user_label = Label(text=f"{user[0]} - {user[1]}", size_hint_x=0.8)
            approve_button = Button(text='Approve', size_hint_x=0.2)
            approve_button.bind(on_press=lambda x, u=user[0]: approve_user(app_instance, u))
            user_layout.add_widget(user_label)
            user_layout.add_widget(approve_button)
            users_layout.add_widget(user_layout)

    scroll_view.add_widget(users_layout)
    content.add_widget(scroll_view)

    app_instance.approve_users_popup = Popup(title='Approve Users', content=content, size_hint=(0.8, 0.5))
    app_instance.approve_users_popup.open()

def approve_user(app_instance, username):
    db_approve_user(username)
    app_instance.approve_users_popup.dismiss()
    show_approve_users_popup(app_instance)

def update_camera_frame(app_instance):
    if app_instance.cap is None:
        return

    ret, frame = app_instance.cap.read()
    if not ret:
        return

    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    app_instance.detector_net.setInput(blob)
    detections = app_instance.detector_net.forward()

    detected_real = False
    name = 'Unknown'

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

            if len(encodings) > 0:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(app_instance.encode_list_known, encoding)
                face_dis = face_recognition.face_distance(app_instance.encode_list_known, encoding)

                if len(face_dis) > 0:
                    best_match_index = np.argmin(face_dis)
                    if matches[best_match_index] and face_dis[best_match_index] < 0.4:
                        name = app_instance.classNames[best_match_index]

            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = app_instance.liveness_model.predict(face)[0]
            label_name = 'real' if np.argmax(preds) == 1 else 'fake'

            if label_name == 'real':
                detected_real = True
                if name == 'Unknown':
                    if app_instance.recognition_start_time is None:
                        app_instance.recognition_start_time = datetime.now()
                    elif datetime.now() - app_instance.recognition_start_time >= timedelta(seconds=5):
                        app_instance.is_unknown_for_5_seconds = True
                        app_instance.capture_btn.disabled = False
                else:
                    app_instance.recognition_start_time = None
                    app_instance.is_unknown_for_5_seconds = False
                    app_instance.capture_btn.disabled = True
                app_instance.result_label.text = f'{name}, {label_name}'

                if label_name == 'fake':
                    cv2.putText(frame, "Đừng cố giả mạo!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                cv2.putText(frame, label_name, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
            else:
                app_instance.recognition_start_time = None
                app_instance.is_unknown_for_5_seconds = False
                app_instance.capture_btn.disabled = True

    if not detected_real:
        app_instance.recognition_start_time = None
        app_instance.is_unknown_for_5_seconds = False
        app_instance.capture_btn.disabled = True

    buf = cv2.flip(frame, 0).tobytes()
    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    app_instance.capture_image_widget.texture = texture
