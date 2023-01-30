
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)

model = keras.models.load_model('/home/user/Desktop/ML/robot_1_full')  # ("final_model_2")
print(model.summary())
