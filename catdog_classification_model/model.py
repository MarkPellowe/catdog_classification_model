import os
import requests
import torch
from torchvision import transforms
import numpy as np
import net
import urllib.request 
import torchvision

CLASS_LABELS = ["cat", "dog"]


class catdog_classification_model:
    def __init__(self):
        ### TODO ###
        ### Download the file containing the weights of the pre-trained model from url
        ### if the file doesn't exist already locally, and write it to a file.
        ### After that, create an instance of the model class.
        ### The code to load the model weights from file and evaluate the model is
        ### already provided.
        filename = "conv_net_model3.ckpt"
        if not os.path.exists(filename):
            file = "https://connectionsworkshop.blob.core.windows.net/pets/conv_net_model3.ckpt"
            urllib.request.urlretrieve(file, filename)
        
        self.model = net.CatDogClassifier()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()


    def predict(self, image, return_props=False):
        # pre-process input image (as required by model)
        transform_input = transforms.Compose([transforms.Resize((256, 256)),])
        image = image.values
        image = image[:, :, 0:3]  # make sure we have only 3 channels
        image = image / 255  # min/max normalisation
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)
        image = image.unsqueeze(0)
        image = transform_input(image)

        # make prediction
        ### TODO ###
        ### Apply the model to the input image.
        ### Find the class with a higher output, and, using the list of
        ### CLASS_LABELS, get the corresponding class name.
        class_probs = self.model(image)
        if not return_probs:
            return(torch.argmax(class_probs)
        else:
            return(class_probs)
        ### Bonus: If you also want the predicted probability for each class,
        ### convert the logits into a probability.


 catdog_classification_model()
