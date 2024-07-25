from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout

def show_admin_dashboard(app_instance):
    app_instance.main_layout.clear_widgets()
    top_bar = BoxLayout(size_hint_y=None, height='48dp')
    logout_button = Button(text='Logout', size_hint_x=None, width='100dp', on_press=app_instance.logout)
    top_bar.add_widget(logout_button)
    app_instance.main_layout.add_widget(top_bar)
    bottom_bar = BoxLayout(size_hint_y=None, height='48dp')
    approve_users_button = Button(text='Xét duyệt users', size_hint_x=None, width='150dp', on_press=app_instance.show_approve_users_popup)
    attendance_button = Button(text='Quản lý chấm công', size_hint_x=None, width='150dp', on_press=app_instance.show_attendance_records)
    manage_users_button = Button(text='Quản lý users', size_hint_x=None, width='150dp', on_press=app_instance.show_manage_users_popup)

    bottom_bar.add_widget(approve_users_button)
    bottom_bar.add_widget(attendance_button)
    bottom_bar.add_widget(manage_users_button)

    app_instance.main_layout.add_widget(BoxLayout())
    app_instance.main_layout.add_widget(bottom_bar)

def show_approve_users_popup(app_instance, instance=None):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    scroll_view = ScrollView()
    users_layout = BoxLayout(orientation='vertical', size_hint_y=None)
    users_layout.bind(minimum_height=users_layout.setter('height'))

    for user in app_instance.get_unapproved_users():
        user_layout = BoxLayout(size_hint_y=None, height=40)
        user_label = Label(text=f"{user[0]} - {user[1]}", size_hint_x=0.8)
        approve_button = Button(text='Duyệt', size_hint_x=0.2)
        approve_button.bind(on_press=lambda x, u=user[0]: app_instance.approve_user(u))
        user_layout.add_widget(user_label)
        user_layout.add_widget(approve_button)
        users_layout.add_widget(user_layout)

    scroll_view.add_widget(users_layout)
    content.add_widget(scroll_view)

    app_instance.approve_users_popup = Popup(title='Duyệt Users', content=content, size_hint=(0.8, 0.5))
    app_instance.approve_users_popup.open()

def show_attendance_records(app_instance, instance=None):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    scroll_view = ScrollView()
    grid_layout = GridLayout(cols=2, size_hint_y=None)
    grid_layout.bind(minimum_height=grid_layout.setter('height'))

    # Thêm tiêu đề cho cột
    grid_layout.add_widget(Label(text="Username", bold=True, size_hint_y=None, height=40))
    grid_layout.add_widget(Label(text="Time", bold=True, size_hint_y=None, height=40))

    for record in app_instance.get_attendance_records():
        username_label = Label(text=record[0], size_hint_y=None, height=40)
        time_label = Label(text=record[1], size_hint_y=None, height=40)
        grid_layout.add_widget(username_label)
        grid_layout.add_widget(time_label)

    scroll_view.add_widget(grid_layout)
    content.add_widget(scroll_view)

    app_instance.attendance_records_popup = Popup(title='Lịch sử điểm danh', content=content, size_hint=(0.8, 0.5))
    app_instance.attendance_records_popup.open()

def show_manage_users_popup(app_instance, instance=None):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    scroll_view = ScrollView()
    users_layout = BoxLayout(orientation='vertical', size_hint_y=None)
    users_layout.bind(minimum_height=users_layout.setter('height'))

    for user in app_instance.get_users():
        if user[2] == 1:  # Only show approved users
            user_layout = BoxLayout(size_hint_y=None, height=40)
            user_label = Label(text=f"{user[0]} - {user[1]}", size_hint_x=0.6)
            delete_button = Button(text='Delete', size_hint_x=0.2)
            delete_button.bind(on_press=lambda x, u=user[0]: app_instance.delete_user(u))
            user_layout.add_widget(user_label)
            user_layout.add_widget(delete_button)
            users_layout.add_widget(user_layout)

    scroll_view.add_widget(users_layout)
    content.add_widget(scroll_view)

    app_instance.manage_users_popup = Popup(title='Quản lý Users', content=content, size_hint=(0.8, 0.5))
    app_instance.manage_users_popup.open()
