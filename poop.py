import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Загрузка изображения
img = load_img('G:/dataset/dress/2adf987d-a985-4023-81d4-adf9368ca251.jpg', target_size=(90, 120))
# Преобразование изображения в массив numpy
img_array = img_to_array(img)
# Нормализация значений пикселей
img_array = img_array / 255.0
# Добавление измерения пакета для массива
img_array = np.expand_dims(img_array, axis=0)

# Загрузка модели Keras
model = tf.keras.models.load_model('G:/clothes_9_class_epoc_30_size_90x120_643.keras')

# Предсказание класса изображения
prediction = model.predict(img_array)

# Отображение диаграммы вероятности для каждого класса
plt.bar(range(len(prediction[0])), prediction[0])
plt.xticks(range(len(prediction[0])))
plt.show()
