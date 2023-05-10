"""
@Project ：.ProjectCode 
@File    ：reg_and_log
@Describe：用户可视化界面
@Author  ：KlNon
@Date    ：2023/5/9 22:23 
"""
import json
import tkinter as tk
from tkinter import ttk

import pymysql
from pathlib import Path
from tkinter import *
import tkinter.messagebox as messagebox
import hashlib
from tkinter import DISABLED

import torch

from visual.scan_pic import get_prediction, load_model
from pytorch.model.label.model_load_label import load_labels
from pytorch.model.model_init import initialize_model

model1, model2, model3 = load_model()  # 1是种类识别模型，2是病虫害识别，3是干枯识别

image_datasets1 = initialize_model(which_file='Kind',
                                   which_model='checkpoint',
                                   output_size=103,
                                   return_params=['image_datasets'])
image_datasets2 = initialize_model(which_file='Diseases',
                                   which_model='checkpoint1',
                                   output_size=8,
                                   return_params=['image_datasets'])
image_datasets3 = initialize_model(which_file='Water',
                                   which_model='checkpoint2',
                                   output_size=4,
                                   return_params=['image_datasets'])

cat_label_to_name1, class_to_idx1 = load_labels(image_datasets1[0], file_name='kind_cat_to_name.json')
cat_label_to_name2, class_to_idx2 = load_labels(image_datasets2[0], file_name='diseases_cat_to_name.json')
cat_label_to_name3, class_to_idx3 = load_labels(image_datasets3[0], file_name='water_cat_to_name.json')


def main():
    def save_as_jpg(img_path):
        from PIL import Image
        img = Image.open(img_path)
        new_path = Path(img_path).with_suffix('.jpg')
        img.save(new_path, 'JPEG')
        return new_path

    def connect_to_db():
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='If=error1977',
            database='model_database',
        )
        return connection

    connection = connect_to_db()

    def on_closing():
        window.destroy()
        connection.close()

        # Function for submitting the registration data

    def submit_register_data():
        def register_with_permissions():
            user = entry_username.get()
            pwd = entry_password.get()
            permissions = permission_var.get()

            if user and pwd and permissions:
                hashed_pwd = hashlib.md5(pwd.encode()).hexdigest()
                try:
                    with connection.cursor() as cursor:
                        sql = "INSERT INTO `User` (`user_name`, `user_password`, `user_permissions`) VALUES (%s, %s, %s)"
                        cursor.execute(sql, (user, hashed_pwd, permissions))
                    connection.commit()
                    messagebox.showinfo("成功", "注册成功")
                except Exception as e:
                    print(e)
                    messagebox.showerror("错误", "注册失败")
            else:
                messagebox.showerror("错误", "请填写所有信息")
            permissions_window.destroy()

        permissions_window = tk.Toplevel(window)
        permissions_window.title("选择权限组")
        permissions_window.geometry("230x150")
        permission_var = tk.StringVar()
        permission_var.set("user")

        tk.Label(permissions_window, text="选择权限组：").grid(row=0, column=0, padx=5, pady=5)
        tk.Radiobutton(permissions_window, text="管理员", variable=permission_var, value="admin").grid(row=1, column=0,
                                                                                                       padx=5, pady=5)
        tk.Radiobutton(permissions_window, text="普通用户", variable=permission_var, value="user").grid(row=1, column=1,
                                                                                                        padx=5,
                                                                                                        pady=5)
        tk.Button(permissions_window, text="确认注册", command=register_with_permissions).grid(row=2, columnspan=2,
                                                                                               pady=10,
                                                                                               sticky='se')

    def open_success_window(user_name, user_permissions):
        from tkinter import filedialog
        from PIL import Image, ImageTk

        def update_progress_bar(value):
            progress_bar['value'] = value
            progress_bar.update_idletasks()

        def update_preview_image(img_path, img_label):
            update_progress_bar(0)  # Reset progress bar
            img = Image.open(img_path)
            update_progress_bar(10)
            img.thumbnail((400, 400))  # 您可以更改此值以调整预览图片的最大尺寸
            img_preview = ImageTk.PhotoImage(img)
            update_progress_bar(30)
            img_label.config(image=img_preview)
            img_label.image = img_preview

        def insert_image_info(pic_path, pic_ann):
            # 数据库连接信息，请根据需要修改
            db_host = "localhost"
            db_user = "root"
            db_password = "If=error1977"
            db_name = "model_database"

            # 连接到MySQL数据库
            connection = pymysql.connect(host=db_host,
                                         user=db_user,
                                         password=db_password,
                                         database=db_name)

            # 使用cursor()方法创建一个游标对象cursor
            cursor = connection.cursor()

            # 插入图像信息
            insert_sql = f"INSERT INTO GetPic (pic_path, pic_ann) VALUES ('{pic_path}', '{pic_ann}');"

            try:
                # 执行SQL语句
                cursor.execute(insert_sql)
                # 提交事务
                connection.commit()
                print("图像信息插入成功")
            except Exception as e:
                # 如果发生错误，则回滚
                connection.rollback()
                print(f"图像信息插入失败，错误：{e}")

            # 关闭游标和数据库连接
            cursor.close()
            connection.close()

        def upload_flower_image():
            filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
            if filename:
                print(f"上传花朵图像: {filename}")

                jpg_path = save_as_jpg(filename)
                update_preview_image(jpg_path, flower_image)
                img = Image.open(jpg_path)
                update_progress_bar(50)
                log_probs, classes = get_prediction(img, model1)
                probs = torch.exp(log_probs.data.cpu()) * 100
                classes = classes.data.cpu()  # 将类别移回CPU
                prob = round(probs[0].squeeze().item(), 2)  # 获取当前图像的概率
                clazz = [cat_label_to_name1[c.item()].title() for c in classes[0]]  # 获取当前图像的类别名称
                update_progress_bar(100)
                print(f"模型预测: {prob, clazz}")  # 此处将输出模型的预测结果

                prediction_info.config(state=NORMAL)
                prediction_info.delete(1.0, END)
                flower_prediction = {
                    "种类模型预测": {"概率": prob, "类别": clazz}
                }
                prediction_info.insert(INSERT, json.dumps(flower_prediction, ensure_ascii=False, indent=2))
                prediction_info.config(state=DISABLED)

                pic_ann = json.dumps(flower_prediction, ensure_ascii=False)
                insert_image_info(jpg_path, pic_ann)

                print(f"将花朵图像另存为: {jpg_path}")
                return flower_prediction

        def upload_leaf_image():
            filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
            if filename:
                # Add your custom code here to process the image file
                # For example: upload the image to the server or save it locally
                # You can use the 'filename' variable to get the file path
                print(f"上传叶片图像: {filename}")

                update_progress_bar(50)
                jpg_path = save_as_jpg(filename)

                # Update leaf image preview
                load_image = Image.open(jpg_path)
                load_image.thumbnail((200, 200))

                log_probs, classes = get_prediction(load_image, model2)
                log_probs2, classes2 = get_prediction(load_image, model3)

                prob = round((torch.exp(log_probs.data.cpu()) * 100)[0].squeeze().item(), 2)  # 获取当前图像的概率
                classes = classes.data.cpu()  # 将类别移回CPU
                clazz = [cat_label_to_name2[c.item()].title() for c in classes[0]]  # 获取当前图像的类别名称

                prob2 = round((torch.exp(log_probs2.data.cpu()) * 100)[0].squeeze().item(), 2)  # 获取当前图像的概率
                classes2 = classes2.data.cpu()  # 将类别移回CPU
                clazz2 = [cat_label_to_name3[c.item()].title() for c in classes2[0]]  # 获取当前图像的类别名称

                print(f"模型预测: {prob, clazz, prob2, clazz2}")  # 此处将输出模型的预测结果
                prediction_info.config(state=NORMAL)
                leaf_prediction = {
                    "病虫害模型预测": {"概率": prob, "类别": clazz},
                    "缺水程度模型预测": {"概率": prob2, "类别": clazz2}
                }

                prediction_info.insert(INSERT, json.dumps(leaf_prediction, ensure_ascii=False, indent=2))
                prediction_info.config(state=DISABLED)
                render = ImageTk.PhotoImage(load_image)
                leaf_image.config(image=render)
                leaf_image.image = render

                update_progress_bar(100)

                pic_ann = json.dumps(leaf_prediction, ensure_ascii=False)
                insert_image_info(jpg_path, pic_ann)

                print(f"将叶片图像另存为: {jpg_path}")
                return leaf_prediction

        window_success = Toplevel(window)
        window_success.title("用户界面")
        window_success.geometry("800x600")

        window.withdraw()
        # Create progress bar for recognition
        progress_bar = ttk.Progressbar(window_success, orient=HORIZONTAL, length=800, mode='determinate')
        progress_bar.pack(side=TOP, pady=5, fill=X)

        # Create frames
        left_frame = Frame(window_success)
        right_frame = Frame(window_success)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Create label, listbox, and scrollbar for left side
        left_label = Label(left_frame, text="花朵图像:")
        left_label.pack(pady=10)
        left_listbox = Listbox(left_frame)
        left_listbox.pack(fill=BOTH, expand=True)
        # 在左侧框架中添加预测信息文本框
        prediction_info = Text(left_frame, wrap=WORD, state=DISABLED)
        prediction_info.pack(fill=BOTH, expand=True)

        # Create labels and buttons for right side
        right_label_user = Label(right_frame, text=f"用户名: {user_name}")
        right_label_permissions = Label(right_frame, text=f"权限组: {user_permissions}")
        right_label_leaf = Label(right_frame, text="叶片图像预览:")
        flower_image = Label(left_listbox)
        leaf_image = Label(right_frame)
        button_upload_flower = Button(right_frame, text="上传花朵图像", command=upload_flower_image)
        button_upload_leaf = Button(right_frame, text="上传叶片图像", command=upload_leaf_image)

        # Place right side labels and buttons
        right_label_user.pack(pady=10)
        right_label_permissions.pack(pady=5)
        flower_image.pack(side=TOP, pady=5)
        button_upload_flower.pack(pady=5)
        right_label_leaf.pack(pady=10)
        leaf_image.pack(pady=5)
        button_upload_leaf.pack(pady=5)

    # Function for verifying the login data
    def verify_login_data():
        user = entry_username.get()
        pwd = entry_password.get()
        hashed_pwd = hashlib.md5(pwd.encode()).hexdigest()

        try:
            with connection.cursor() as cursor:
                sql = "SELECT `user_permissions` FROM `User` WHERE `user_name`=%s AND `user_password`=%s"
                cursor.execute(sql, (user, hashed_pwd))
                result = cursor.fetchone()
                if result:
                    messagebox.showinfo("成功", f"登录成功！\n权限组：{result[0]}")
                    open_success_window(user, result[0])
                else:
                    messagebox.showerror("错误", "用户名或密码错误")
        except Exception as e:
            print(e)
            messagebox.showerror("错误", "登录失败")

    # Create the main window
    window = Tk()
    window.title("Login and Register")
    window.geometry("300x200")

    # Set the weight for rows and columns
    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    window.rowconfigure(2, weight=1)
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)

    # Create labels and entry widgets for the username and password
    label_username = Label(window, text="用户名:")
    label_password = Label(window, text="密码:")
    entry_username = Entry(window)
    entry_password = Entry(window, show="*")

    # Place the labels and entry widgets using grid layout
    label_username.grid(row=0, column=0, padx=10, pady=(10, 0), sticky='e')
    entry_username.grid(row=0, column=1, padx=10, pady=(10, 0), sticky='w')
    label_password.grid(row=1, column=0, padx=10, pady=(10, 0), sticky='e')
    entry_password.grid(row=1, column=1, padx=10, pady=(10, 0), sticky='w')

    # Create login and register buttons
    button_login = Button(window, text="登录", command=verify_login_data)
    button_register = Button(window, text="注册", command=submit_register_data)

    # Place the buttons at the bottom left and right
    button_login.grid(row=3, column=0, pady=10, sticky='sw')
    button_register.grid(row=3, column=1, pady=10, sticky='se')

    # Run the main event loop
    window.mainloop()


if __name__ == "__main__":
    main()
