"""
@Project ：.ProjectCode 
@File    ：uploadPic
@Describe：
@Author  ：KlNon
@Date    ：2023/5/10 2:20 
"""
import pymysql

# 连接到MySQL服务器
connection = pymysql.connect(
    host='localhost',  # 替换为您的数据库服务器地址
    user='root',  # 替换为您的数据库用户名
    password='If=error1977',  # 替换为您的数据库密码
    charset='utf8mb4'
)

# 创建一个游标对象，用于执行SQL命令
cursor = connection.cursor()

# 创建数据库
cursor.execute("CREATE DATABASE IF NOT EXISTS model_database;")

# 使用数据库
cursor.execute("USE model_database;")

# 创建数据表
create_table_query = """
CREATE TABLE IF NOT EXISTS GetPic (
    pic_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
    pic_ann VARCHAR(255),
    pic_path VARCHAR(255)
);
"""

cursor.execute(create_table_query)

# 提交更改并关闭连接
connection.commit()
cursor.close()
connection.close()
