import numpy as np
import tensorflow as tf
from keras.src.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, F1Score
import gradio as gr
import cv2

label_mapping = {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}

# Load Data
data = r'D:\Dataset'
dataset = tf.keras.utils.image_dataset_from_directory(data,label_mode='categorical')

data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()

# Scale Data
data = dataset.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# Train Test Split
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Model
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Extract target labels
train_labels = np.array([y for _, y in train.as_numpy_iterator()])
val_labels = np.array([y for _, y in val.as_numpy_iterator()])

# Convert target labels to one-hot encoded vectors
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

# Train
hist = model.fit(train, epochs=10, validation_data=val)

loss,accuracy = model.evaluate(test)
print(f'Loss: {loss}, Accuracy: {accuracy * 100}')

model.save('AlzheimerCNN.h5')
with open('training_history.npy', 'wb') as f:
    np.save(f, hist.history)
# Loss
def plot_loss():
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

# Accuracy
def plot_accuracy():
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

plot_loss()
plot_accuracy()

# Precision, Recall and F1 Score
pre = Precision()
re = Recall()
f1 = F1Score()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    f1.update_state(y, yhat)

print(f'Precision: {pre.result()}, Recall: {re.result()}, F1: {f1.result()}')


# Gradio interface
def predict_class(image):
    img = cv2.resize(image, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    if predicted_class in label_mapping:
        return {label_mapping[predicted_class]: float(prediction[0][predicted_class])}
    else:
        return {"Unknown Class": predicted_class}

iface = gr.Interface(predict_class, inputs="image", outputs="label", title="Image Classification")
iface.launch(share=True)



