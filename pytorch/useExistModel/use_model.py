"""
@Project ：.ProjectCode 
@File    ：use_model
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 22:42
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from pytorch.model.data_config import gdrive_dir, device, dataloaders
from pytorch.model.init import load_checkpoint
from pytorch.model.label.model_load_label import cat_label_to_name, class_to_idx
from pytorch.model.test.model_test import modelTest
from pytorch.model.data_config import data_dir
from PIL import Image

from pytorch.useExistModel.data_prepare.data_prepare import process_image
from pytorch.useExistModel.img.img_show import imshow
from pytorch.useExistModel.predict.model_predict import predict

if __name__ == '__main__':
    model = load_checkpoint(gdrive_dir + 'checkpoint.pt')

    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['test_data']:

            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Get predictions for this test batch
            output = model(images)

            output = torch.exp(output)
            probs, classes = output.topk(5, dim=1)

            # Display an image along with the top 5 classes

            images = images.cpu()
            probs = probs.data.cpu()
            classes = classes.data.cpu()
            fig = plt.figure(figsize=(15, 28),
                             tight_layout={'pad': 0, 'w_pad': 5, 'h_pad': 0, 'rect': (0, 0, 1, 1)})
            rows = 4
            lines = 16
            line_ctrl = 0
            for index in range(len(images)):
                if index % rows == 0:
                    line_ctrl += 1

                prob = probs[index].squeeze()

                clazz = [cat_label_to_name[c.item()].title() for c in classes[index]]

                label = labels[index].item()
                title = f'{cat_label_to_name[label].title()}'

                position = index + 1 + (line_ctrl * rows)
                ax1 = fig.add_subplot(lines, rows, position, xticks=[], yticks=[])
                titlecolor = 'g'
                if title != clazz[0]:
                    titlecolor = 'r'

                imshow(images[index], ax1, title, titlecolor=titlecolor)

                ax2 = fig.add_subplot(lines, rows, position + rows, xticks=[], yticks=[])
                ax2.barh(np.arange(5), prob)
                ax2.set_yticks(np.arange(5))
                ax2.set_yticklabels(clazz)
                ax2.set_ylim(-1, 5)
                ax2.invert_yaxis()
                ax2.set_xlim(0, 1.1)

            # break
            plt.show(block=True)

    modelTest(model, show_graphs=True)
