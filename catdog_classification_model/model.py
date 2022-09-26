import os
import requests
import torch
from torchvision import transforms
import numpy as np
import net

CLASS_LABELS = ["cat", "dog"]


class CatDogClassifier:
    def __init__(self):
        ### TODO ###
        ### Download the file containing the weights of the pre-trained model from url
        ### if the file doesn't exist already locally, and write it to a file.
        ### After that, create an instance of the model class.
        ### The code to load the model weights from file and evaluate the model is
        ### already provided.
        filename = "conv_net_model3.ckpt"
        if not os.path.exists(filename):
            model_path = os.path.join(
                "https://connectionsworkshop.blob.core.windows.net/pets", filename
            )
            r = requests.get(model_path)
            with open(filename, "wb") as outfile:
                outfile.write(r.content)
        self.model = net.CatDogClassifier()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(self, image):
        # pre-process input image (as required by model)
        transform_input = transforms.Compose([transforms.Resize((256, 256)),])
        image = image.values
        image = image[:, :, 0:3]  # make sure we have only 3 channels
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)
        image = image.unsqueeze(0)
        image = transform_input(image)

        # make prediction
        ### TODO ###
        ### Apply the model to the input image.
        ### Find the class with a higher output, and, using the list of
        ### CLASS_LABELS, get the corresponding class name.
        ### Bonus: If you also want the predicted probability for each class,
        ### convert the logits into a probability.

        return "FIX ME"

