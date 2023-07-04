from django.shortcuts import render
import os
import cv2
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError

from django.core.files.storage import FileSystemStorage

import glob

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


# Create your views here.
def home(request):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'NULL', 'a', 'b', 'bye', 'c', 'd', 'e', 'good', 'good morning', 'hello', 'little bit', 'no', 'pardon', 'please', 'project', 'whats up', 'yes']
    message = ""
    prediction = ""
    img_shape=128
    fss = CustomFileSystemStorage()
    try:
        image = request.FILES["image"]
        print(image)
        file_type = image.content_type
        if file_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff", "image/svg+xml"]:
            return TemplateResponse(
            request,
            "home.html",
            {"message": "Wybrano niepoprawny rodzaj danych, wybierz zdjęcie"},
            )
        print("Name", image.file)
        print("Type file ", image.content_type)

        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        # image details
        image_url = fss.url(_image)

        # Read the image
        imag = tf.io.read_file(path)
        img_from_ar = tf.image.decode_image(imag, channels=3)
        resized_image = tf.image.resize(img_from_ar, size = [img_shape, img_shape])
        resized_image = resized_image/255.
        model = tf.keras.models.load_model(os.getcwd() + '/model4.h5')
        pred = model.predict(tf.expand_dims(resized_image, axis=0))
        result = round (100 * np.amax(pred), 4)
        result = str(result) + "%"

        pred_class = class_names[pred.argmax()] # if more than one output, take the max
        print(pred)
        print("Prediction: " + str(np.argmax(pred)))
        prediction = pred_class 
        
        return TemplateResponse(
            request,
            "home.html",
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": prediction,
                "score": result,
            },
        )
    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "home.html",
            {"message": "Nie wybrano zdjęcia"},
        )


def about(request):
    # return render(request, 'about.html')
    images = []
    
    images_files = glob.glob(os.path.join('static', 'dataset', '*.jpeg'))  # Wczytaj wszystkie pliki JPEG z folderu static/datas

    for image_file in images_files:
        filename = os.path.basename(image_file)
        alt = os.path.splitext(filename)[0]
        title = alt.capitalize()

        image_data = {
            'filename': filename,
            'alt': alt,
            'title': title,
        }

        images.append(image_data)

    context = {
        'images': images,
    }

    return render(request, 'about.html', context)

def contact(request):
    return render(request, 'contact.html')