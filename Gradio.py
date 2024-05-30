import tensorflow as tf
import cv2
import numpy as np
import gradio as gr
from matplotlib import pyplot as plt


CNN = tf.keras.models.load_model('AlzheimerCNN.h5')

with open('training_history.npy', 'rb') as f:
    hist = np.load(f, allow_pickle=True).item()

# Plotting Loss
def plot_loss():
    fig = plt.figure()
    plt.plot(hist['loss'], color='teal', label='loss')
    plt.plot(hist['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

# Plotting Accuracy
def plot_accuracy():
    fig = plt.figure()
    plt.plot(hist['accuracy'], color='teal', label='accuracy')
    plt.plot(hist['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

# Plot the loss and accuracy
plot_loss()
plot_accuracy()

label_mapping = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}


# Gradio interface
def predict_class(image):
    img = cv2.resize(image, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = CNN.predict(img)
    predicted_class = np.argmax(prediction)
    if predicted_class in label_mapping:
        return {label_mapping[predicted_class]: float(prediction[0][predicted_class])}
    else:
        return {"Unknown Class": predicted_class}
    #return {predicted_class: float(prediction[0][predicted_class])}

iface = gr.Interface(predict_class, inputs="image", outputs="label", title="Image Classification")
iface.launch()