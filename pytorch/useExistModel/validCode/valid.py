"""
@Project ：.ProjectCode 
@File    ：valid
@Describe：
@Author  ：KlNon
@Date    ：2023/4/25 15:49 
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from pytorch.model.label.model_load_label import load_labels
from pytorch.model.model_init import load_checkpoint, initialize_model
from pytorch.useExistModel.img.img_show import imshow
from pytorch.useExistModel.predict.model_predict import predict
from PIL import Image
from pytorch.useExistModel.data_prepare.data_prepare import process_image

if __name__ == '__main__':
    checkpoint_dir, data_dir, device, image_datasets, dataloaders = initialize_model(which_file='Kind',
                                                                                     which_model='checkpoint',
                                                                                     output_size=103,
                                                                                     return_params=['checkpoint_dir',
                                                                                                    'data_dir',
                                                                                                    'device',
                                                                                                    'image_datasets',
                                                                                                    'dataloaders'])
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='kind_cat_to_name.json')
    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')

    model.to(device)

    # modelTest(model, dataloaders['test_data'], device, show_graphs=True)
    #
    category = 30
    image_name = 'image_08077.jpg'
    image_path = data_dir + f'/valid/{category}/{image_name}'
    #
    # image = Image.open(image_path)
    # image = process_image(image)
    # imshow(image)
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)

    # Display an image along with the top 5 classes
    probs = probs.data.cpu()
    probs = probs.numpy().squeeze()

    classes = classes.data.cpu()
    classes = classes.numpy().squeeze()
    classes = [cat_label_to_name[clazz].title() for clazz in classes]

    label = class_to_idx[str(category)]
    title = f'{cat_label_to_name[label].title()}'

    fig = plt.figure(figsize=(4, 10))

    ax1 = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])

    image = Image.open(image_path)
    image = process_image(image)
    imshow(image, ax1, title)

    ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
    ax2.barh(np.arange(5), probs)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(classes)
    ax2.set_ylim(-1, 5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Class Probability')

    plt.show()
