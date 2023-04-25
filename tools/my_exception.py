"""
@Project ：.ProjectCode 
@File    ：my_exception
@Describe：学习异常时候用到的
@Author  ：KlNon
@Date    ：2023/3/13 22:55 
"""


class MyException(Exception):
    def __init__(self, message):
        self.message = message


try:
    raise MyException("这是一个自定义异常")
except MyException as e:
    print("捕获到异常:", e.message)
