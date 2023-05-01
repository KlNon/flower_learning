"""
@Project ：.ProjectCode 
@File    ：use_model
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 22:42
"""
from matplotlib import pyplot as plt

from pytorch.model.label.model_load_label import load_labels
from pytorch.model.model_init import *
from pytorch.model.test.model_test import modelTest
from pytorch.useExistModel.img.img_show import imshow

checkpoint_dir, data_dir, device, model, data_transforms, image_datasets, dataloaders, data_classes = initialize_model()
cat_label_to_name, class_to_idx = load_labels(image_datasets)


def plot_bar(ax_bar, prob, class_indices):
    class_names = [cat_label_to_name[c.item()].title() for c in class_indices]
    ax_bar.barh(np.arange(5), prob)
    ax_bar.set_yticks(np.arange(5))
    ax_bar.set_yticklabels(class_names)
    ax_bar.set_ylim(-1, 5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.1)


def main():
    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in dataloaders['test_data']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            output = torch.exp(output)
            probs, classes = output.topk(5, dim=1)

            images = images.cpu()
            probs = probs.data.cpu()
            classes = classes.data.cpu()

            fig, axes = plt.subplots(16, 8, figsize=(15, 28),
                                     tight_layout={'pad': 0, 'w_pad': 5, 'h_pad': 0, 'rect': (0, 0, 1, 1)})

            for index, (image, prob, class_indices, label) in enumerate(zip(images, probs, classes, labels)):
                row = index % 4
                line = index // 4

                true_label = cat_label_to_name[label.item()].title()
                title_color = 'g' if true_label == cat_label_to_name[class_indices[0].item()].title() else 'r'

                ax_image = axes[line * 2, row]
                imshow(image, ax_image, true_label, title_color)

                ax_bar = axes[line * 2 + 1, row]
                plot_bar(ax_bar, prob, class_indices)

            plt.show()

    modelTest(model, show_graphs=True)


if __name__ == '__main__':
    main()
