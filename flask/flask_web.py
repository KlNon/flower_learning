"""
@Project ：.ProjectCode
@File    ：flask
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 16:41
"""
import os

from flask import Flask, request, render_template
from pytorch.model.net.model_net import Net
from PIL import Image
import torchvision.transforms as transforms
import torch

app = Flask(__name__)

'''
卷积神经网络相关程序
'''

# 数据预处理
data_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # 数据归一化
])

# 类别
data_classes = ('丰花月季', '地被月季', '壮花月季', '大花香水月季', '微型月季', '树状月季', '灌木月季', '藤本月季')

# 选择CPU还是GPU的操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 选择模型
net = Net()
net.load_state_dict(torch.load('../flower_net.pth'))

net.eval()
net.to(device)

'''
flask相关程序
'''


def get_prediction_results(img_list):
    results = []
    for img_path in img_list:
        # 获取原始类别
        original_class = os.path.basename(os.path.dirname(img_path))
        # 加载图像
        img = Image.open(img_path)
        # 预处理图像
        img = data_transform(img)  # 这里经过转换后输出的 input 格式是 [C,H,W]，网络输入还需要增加一维批量大小B
        img = img.unsqueeze(0)  # 增加一维，输出的 img 格式为 [1,C,H,W]
        img = img.to(device)
        # 进行预测
        with torch.no_grad():
            outputs = net(img)
            _, preds = torch.max(outputs, 1)
            score = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()
            predicted_class = data_classes[preds[0]]
        is_tp = original_class == predicted_class
        is_fp = original_class != predicted_class and predicted_class in data_classes
        is_fn = original_class != predicted_class and predicted_class not in data_classes
        # 记录预测结果
        result = {
            'img_path': img_path,
            'original_class': original_class,
            'predicted_class': predicted_class,
            'score': score,
            'is_tp': is_tp,
            'is_fp': is_fp,
            'is_fn': is_fn
        }
        results.append(result)
    return results


def prediction_results_to_html_tables(results):
    # 分组
    groups = {}
    for r in results:
        if r['original_class'] not in groups:
            groups[r['original_class']] = []
        groups[r['original_class']].append(r)

    # 生成HTML表格
    html_tables = []
    for group_name, group_results in groups.items():
        html_table = "<table class='prediction-table'>"
        # 表头
        html_table += f"<tr><th colspan='3'>{group_name}</th></tr>"
        html_table += "<tr><th>图片路径</th><th>原始类别</th><th>预测类别</th><th>符合度</th><th>True Positive</th><th>False " \
                      "Positive</th><th>False Negative</th></tr> "
        # 表格内容
        for r in group_results:
            tp_count = sum(r['is_tp'] for r in group_results)
            fp_count = sum(r['is_fp'] for r in group_results)
            fn_count = sum(r['is_fn'] for r in group_results)
            html_table += "<tr>"
            html_table += f"<td>{r['img_path']}</td>"
            html_table += f"<td>{r['original_class']}</td>"
            html_table += f"<td>{r['predicted_class']}</td>"
            html_table += f"<td>{r['score']}</td>"
            if r['is_tp']:
                html_table += "<td>✔</td><td></td><td></td>"
            elif r['is_fp']:
                html_table += "<td></td><td>✔</td><td></td>"
            else:
                html_table += "<td></td><td></td><td>✔</td>"
            html_table += "</tr>"
        html_table += "</table>"
        html_table += f"<tr><td colspan='4'><b>分类准确率：{tp_count / (tp_count + fp_count):.1%}</b></td></tr> "
        html_tables.append(html_table)

    return html_tables


# 定义函数，统计每个类别的TP、FP、FN数量
def count_true_positives(pred_results):
    TP = {class_name: 0 for class_name in data_classes}
    FP = {class_name: 0 for class_name in data_classes}
    FN = {class_name: 0 for class_name in data_classes}
    for r in pred_results:
        if r['original_class'] == r['predicted_class']:
            TP[r['original_class']] += 1
        else:
            FP[r['predicted_class']] += 1
            FN[r['original_class']] += 1
    return TP, FP, FN


# 定义函数，计算每个类别的预测准确率和召回率
# TP表示真正例（True Positive）
# FP表示假正例（False Positive）
# FN表示假反例（False Negative）
# 当进行分类预测时，如果预测结果与实际结果相同，则是真正例（True Positive，TP），例如在癌症筛查中，如果预测结果是该患者患有癌症，实际上该患者真的患有癌症，那么就是真正例。
#
# 如果预测结果为正例但实际结果为负例，则是假正例（False Positive，FP），例如在安检场合中，如果安检人员将一名正常乘客错误认为是恐怖分子，那么就是假正例。
#
# 如果预测结果为负例但实际结果为正例，则是假反例（False Negative，FN），例如在药物治疗中，如果医生将一个患者的病情判断为健康，实际上该患者的病情很严重，那么就是假反例。
#
# 可以用比喻来理解，真正例就像是“命中靶心”，假正例就像是“误伤无辜”，假反例就像是“漏网之鱼”。
def compute_precision_recall(TP, FP, FN):
    precision = {}
    recall = {}

    for class_name in TP:
        if TP[class_name] + FP[class_name] == 0:
            precision[class_name] = 0
        else:
            precision[class_name] = TP[class_name] / (TP[class_name] + FP[class_name])

        if TP[class_name] + FN[class_name] == 0:
            recall[class_name] = 0
        else:
            recall[class_name] = TP[class_name] / (TP[class_name] + FN[class_name])

    return precision, recall


def prediction_TPFPFN_to_html_tables(results):
    # 统计每个类别的TP、FP、FN数量
    TP, FP, FN = count_true_positives(results)
    # 计算每个类别的预测准确率和召回率
    precision, recall = compute_precision_recall(TP, FP, FN)
    table_html = "<table>"
    table_html += "<tr><th>类别</th><th>预测准确率</th><th>召回率</th></tr>"
    for class_name in TP:
        table_html += "<tr>"
        table_html += f"<td>{class_name}</td>"
        precision_str = f"{precision[class_name]:.2%}"
        recall_str = f"{recall[class_name]:.2%}"
        table_html += f"<td>{precision_str}</td>"
        table_html += f"<td>{recall_str}</td>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


@app.route('/inference')
def inference():
    data_folder = request.args.get('url')  # 数据集文件夹路径
    img_list = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # 假设数据集中只包含这两种格式的图片
                img_path = os.path.join(root, file)
                img_list.append(img_path)

    result_list = get_prediction_results(img_list)

    html_tables = prediction_results_to_html_tables(result_list)
    html_tables.append(prediction_TPFPFN_to_html_tables(result_list))
    return render_template("index.html", tables=html_tables)


if __name__ == '__main__':
    app.run()
