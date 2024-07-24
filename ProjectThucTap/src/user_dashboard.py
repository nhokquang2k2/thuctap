from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from database import get_user_image

def show_user_dashboard(app_instance):
    app_instance.main_layout.clear_widgets()
    layout = BoxLayout(orientation='vertical')
    profile_image = app_instance.get_user_profile_image(app_instance.username)
    if profile_image:
        texture = app_instance.convert_image_to_texture(profile_image)
        image_widget = Image(texture=texture, size_hint=(1, None), height=200)
        layout.add_widget(image_widget)
    username_label = Label(text=f"Username: {app_instance.username}", size_hint=(1, None), height=40)
    layout.add_widget(username_label)
    logout_button = Button(text="Logout", on_press=app_instance.logout, size_hint=(1, None), height=40)
    layout.add_widget(logout_button)

    app_instance.main_layout.add_widget(layout)
