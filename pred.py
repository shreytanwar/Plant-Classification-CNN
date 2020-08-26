import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras import models
model = models.load_model('leaf_pred.h5')
model.summary()


path = ('dataset/predict/abies_concolor.jpg')
img = image.load_img(path, target_size=(64, 64))
#img = img.reshape(64,64,1)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
result=model.predict_classes(img)
