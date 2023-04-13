"""
@Project ：.ProjectCode 
@File    ：flower_crawler
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 20:27 
"""
from lxml import etree

import requests
import os
from bs4 import BeautifulSoup

visited_links = set()  # 保存已经访问过的链接
classes = ('藤本月季', '微月', '切花月季', '丰花月季', '大花月季', '冷门月季', '造型月季')


def save_image(img_src, img_name, dir_path):
    if '?' in img_name:  # 取消图片名字中的参数
        img_name = img_name.split('&')[0]

    img_path = os.path.join(dir_path, img_name + '.jpg')  # 文件保存路径

    # 检查文件是否已经存在，如果已经存在则在文件名中添加序号
    if os.path.exists(img_path):
        num = 1
        while True:
            new_url = img_path[:-4] + '_' + str(num) + '.jpg'  # 加上序号并修改后缀
            if not os.path.exists(new_url):
                img_path = new_url
                break
            num += 1

    # 下载图片
    try:
        headers = {'Content-Type': 'image/jpeg'}  # 添加Content-Type头
        with open(img_path, 'wb') as f:
            img = requests.get(img_src, headers=headers).content
            f.write(img)
    except Exception as e:
        print('Failed to save image: {}\nError: {}'.format(img_src, e))


def download_images_from_url(url, base_dir, xpath):
    # 检查这个链接是否已经访问过
    if url in visited_links:
        print('skip：', url)
        return  # 如果访问过了，就直接返回

    visited_links.add(url)

    if not os.path.exists(base_dir):  # 如果文件夹不存在则创建
        os.mkdir(base_dir)

    # 获取HTML代码，并解析出所有的图片链接
    response = requests.get(url, verify=False)
    response.encoding = response.apparent_encoding  # 自动识别网页编码
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all('img')

    parser = etree.HTMLParser()
    root = etree.fromstring(html, parser)
    element = root.xpath(xpath)[0] if root.xpath(xpath) else None  # 在解析后的 HTML 页面中查找元素
    try:
        element_text = element.text
    except Exception as e:
        element_text = ''
        print('Failed to get element text, continue...')

    if element_text in classes:
        dir_name = element_text.strip().replace(' ', '_')  # 将元素文本中的空格替换成下划线，并去掉前后空格
        dir_path = os.path.join(base_dir, dir_name)

        if not os.path.exists(dir_path):  # 如果文件夹不存在则创建
            os.mkdir(dir_path)

        for img_tag in img_tags:
            img_src = img_tag['src']
            if not img_src.startswith('https://www.tengbenyueji.com'):
                img_src = ''.join(['https://www.tengbenyueji.com', '/', img_src])
            img_name = img_tag.get('title')  # 如果没有title属性，则默认以'image'作为标题
            if img_name is None:
                print('No title attribute, continue...')
                continue
            img_name = img_name
            print('Downloading image: {},:{}'.format(img_name, img_src))
            save_image(img_src, img_name, dir_path)

    # 找到当前页面中的所有链接，并递归访问
    try:
        link_tags = soup.find_all('a')
        for link_tag in link_tags:
            link = link_tag.get('href')
            if link.startswith('/'):  # 相对路径拼接成绝对路径
                link = "https://www.tengbenyueji.com" + link
            if link.startswith('https://www.tengbenyueji.com'):  # 只访问目标网站下的链接
                download_images_from_url(link, base_dir, xpath)
    except:
        return


if __name__ == '__main__':
    root_url = "https://www.tengbenyueji.com"
    download_images_from_url(root_url, "roses", '//*[@id="detail-article"]/div[2]/a')
