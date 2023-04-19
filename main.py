import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import utils
from keras.preprocessing import image
from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
data_dir = 'G:\\dataset'
image_size = (90, 120)
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./ 255,# Масштабирование значений пикселей в диапазон [0, 1]
    rotation_range=20,  # Случайный поворот изображения на угол в диапазоне [-20, 20]
    width_shift_range=0.2,  # Случайный сдвиг изображения по горизонтали на долю от ширины изображения
    height_shift_range=0.2,  # Случайный сдвиг изображения по вертикали на долю от высоты изображения
    horizontal_flip=True,  # Случайное отражение изображения по горизонтали
    validation_split=0.2  # Доля изображений для валидации
)

# train_dataset = image_dataset_from_directory(data_dir,
#                                              subset='training',
#                                              seed=42,
#                                              validation_split=0.1,
#                                              batch_size=batch_size,
#                                              image_size=image_size)

# validation_dataset = image_dataset_from_directory(data_dir,
#                                                   subset='validation',
#                                                   seed=42,
#                                                   validation_split=0.1,
#                                                   batch_size=batch_size,
#                                                   image_size=image_size)

# class_names = train_dataset.class_names
# print(class_names)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)
test_dataset = image_dataset_from_directory(data_dir,
                                            batch_size=batch_size,
                                            image_size=image_size)


# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Создаем последовательную модель
model = Sequential()
# Сверточный слой
model.add(Conv2D(16, (5, 5), padding='same',
                 input_shape=(90, 120, 3), activation='relu'))
# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Сверточный слой
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Сверточный слой
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Сверточный слой
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Полносвязная часть нейронной сети для классификации
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# Выходной слой, 131 нейрон по количеству классов
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    verbose=1)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
model.save("G:\\clothes_9_class_model.keras")