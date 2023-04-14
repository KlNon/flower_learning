"""
@Project ：.ProjectCode 
@File    ：flower_crawler
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 20:27 
"""
import re

from lxml import etree

import requests
import os
import time
import urllib3
from bs4 import BeautifulSoup
import sys

sys.setrecursionlimit(1000000)  # 设置递归深度限制为 1000000

visited_links = set()  # 保存已经访问过的链接
classes = ('藤本月季', '微月', '切花月季', '丰花月季', '大花月季', '冷门月季', '造型月季')
MAX_RETRY = 5  # 最大重试次数

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def save_image(img_src, img_name, dir_path):
    if '?' in img_name:  # 取消图片名字中的参数
        img_name = img_name.split('&')[0]

    img_path = os.path.join(dir_path, img_name)  # 文件保存路径

    # 如果img_name中不包含后缀，则添加'.jpg'后缀
    if not img_name.endswith('.jpg'):
        img_path += '.jpg'

    # 检查文件是否已经存在，如果已经存在则在文件名中添加序号
    if os.path.exists(img_path):
        num = 1
        while True:
            new_url = img_path[:-4] + '_' + str(num) + '.jpg'  # 加上序号并修改后缀，去掉原先已经添加的'.jpg'后缀
            if not os.path.exists(new_url):
                img_path = new_url
                break
            num += 1

    # 下载图片
    for i in range(MAX_RETRY):
        try:
            headers = {'Content-Type': 'image/jpeg'}  # 添加Content-Type头
            with open(img_path, 'wb') as f:
                img = requests.get(img_src, headers=headers).content
                if len(img) > 0:  # 判断是否下载到非空图片
                    f.write(img)
                    break  # 成功下载图片，退出循环
        except Exception as e:
            print('Failed to save image: {}\nError: {}'.format(img_src, e))
        print('Retry #{} to download image: {}'.format(i + 1, img_src))
        time.sleep(3)  # 等待3秒后重试


def download_images_from_url(url, base_dir, xpath, xpath2):
    visited_links.add(url)  # 添加到已访问
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # 用一个列表来保存将要访问的链接
    links = [url]

    while links:
        link = links.pop(0)  # 从列表中弹出第一个元素

        # 获取HTML代码，并解析出所有的图片链接
        response = requests.get(link, verify=False)
        response.encoding = response.apparent_encoding
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        img_tags = soup.find_all('img')

        parser = etree.HTMLParser()
        root = etree.fromstring(html, parser)
        element = root.xpath(xpath)[0] if root.xpath(xpath) else None
        try:
            element_text = element.text
        except Exception as e:
            element_text = ''
            # print('Failed to get element text, continue...')

        if element_text in classes:
            dir_name = element_text.strip().replace(' ', '_')
            dir_path = os.path.join(base_dir, dir_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            for img_tag in img_tags:
                img_src = img_tag['src']
                if not img_src.startswith('https://www.tengbenyueji.com'):
                    img_src = ''.join(['https://www.tengbenyueji.com', '/', img_src])
                img_name = img_tag.get('title')
                if img_name is None:
                    continue
                img_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', '', img_name)
                img_name = img_name.replace('jpg', '')
                if img_name == '':
                    img_name = root.xpath(xpath2)[0].text if root.xpath(xpath2) else None
                img_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', '', img_name)
                print('Downloading image: {} :{}'.format(img_name, img_src))
                save_image(img_src, img_name, dir_path)

        # 找到当前页面中的所有链接，并添加到列表中
        try:
            link_tags = soup.find_all('a')
            for link_tag in link_tags:
                link = link_tag.get('href')
                if link.startswith('/'):
                    link = "https://www.tengbenyueji.com" + link
                if link.startswith('https://www.tengbenyueji.com'):
                    if link not in visited_links:
                        links.append(link)
                        visited_links.add(link)
        except Exception as e:
            print("Failed to get links from url: {} ,Error:{}".format(link_tag, e))


if __name__ == '__main__':
    root_url = "https://www.tengbenyueji.com"
    download_images_from_url(root_url, "roses", '//*[@id="detail-article"]/div[2]/a',
                             '//*[@id="detail-article"]/div[1]/h1')
