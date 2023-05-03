"""
@Project ：.ProjectCode 
@File    ：model_predict
@Describe：
@Author  ：KlNon
@Date    ：2023/4/25 15:45 
"""
import torch
from PIL import Image
from pytorch.useExistModel.data_prepare.data_prepare import process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)

    with torch.no_grad():
        model.eval()

        image = image.view(1, 3, 224, 224)
        image = image.to(device)

        predictions = model.forward(image)

        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)

    return top_ps, top_class, predictions
