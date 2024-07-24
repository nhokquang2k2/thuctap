from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from src.database import authenticate_user

def show_login_popup(app_instance):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    app_instance.login_username_input = TextInput(hint_text='Username', multiline=False, size_hint_y=None, height=40)
    app_instance.login_password_input = TextInput(hint_text='Password', multiline=False, password=True, size_hint_y=None, height=40)
    login_btn = Button(text='Login', on_press=app_instance.validate_login, size_hint_y=None, height=40)
    content.add_widget(app_instance.login_username_input)
    content.add_widget(app_instance.login_password_input)
    content.add_widget(login_btn)

    app_instance.login_popup = Popup(title='Login', content=content, size_hint=(0.8, 0.5))
    app_instance.login_popup.open()
    app_instance.login_popup.open()

def validate_login(app_instance, instance):
    username = app_instance.username_input.text
    password = app_instance.password_input.text
    if authenticate_user(username, password):
        app_instance.login_popup.dismiss()
        app_instance.username = username
        app_instance.logged_in_role = 'admin' if username == 'admin' else 'user'
        app_instance.show_dashboard()
    else:
        app_instance.show_notification("Invalid username or password.")
