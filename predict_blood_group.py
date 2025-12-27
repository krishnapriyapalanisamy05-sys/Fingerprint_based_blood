
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = load_model('model_blood_group_detection_alextnet.keras')

# Label mapping
labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

# Set image path (make sure this file exists)
img_path = 'dataset_blood_group/A+/cluster_0_1001.BMP'

# Load and preprocess image
img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Prediction
result = model.predict(x)
predicted_class = np.argmax(result)
predicted_label = labels[predicted_class]
confidence = result[0][predicted_class] * 100

# Show result
plt.imshow(image.array_to_img(image.img_to_array(img) / 255.0))
plt.axis('off')
plt.title(f"Prediction: {predicted_label} with confidence {confidence:.2f}%")
plt.show()
