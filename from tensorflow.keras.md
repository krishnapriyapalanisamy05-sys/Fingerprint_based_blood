from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

def predict_blood_group(model, img):
    img_resized = img.resize((256, 256))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    result = model.predict(x)
    predicted_class = np.argmax(result)
    confidence = result[0][predicted_class] * 100
    return predicted_class, confidence