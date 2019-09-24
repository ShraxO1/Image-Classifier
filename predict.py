import cv2
import tensorflow as tf

Categories=["cat","dog"]

model=tf.keras.models.load_model("model.h6")
img = cv2.imread("18.jpg")
img = cv2.resize(img, (50,50))
img = img.reshape(1, 50, 50, 3)
print(int(model.predict(img)))
prediction=(int(model.predict(img)))
print(Categories[prediction])
