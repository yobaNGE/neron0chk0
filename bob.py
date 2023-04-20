import tensorflow as tf
from tensorflow import keras

# Load the Keras model
model = keras.models.load_model("G:/clothes_9_class_epoc_30_size_90x120_643.keras")
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("G:/models_out/model.tflite", "wb") as f:
    f.write(tflite_model)