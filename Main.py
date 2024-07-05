'''
Entraînement d'un réseau de neurones récurrents pour l'identification de lignes cadentielles 
Jeu de données : Praetorius, Terpsichore (1612)


use keras-2.6.0
tensorboard-2.6.0 
tensorflow-2.6.2  
'''

 
if __name__ == '__main__':
    pass

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

 

''' import dataset '''

features= np.load('./data/observations.npy')
labels = np.load('./data/labels.npy')
ids = np.load('./data/ids.npy')

 

trainTestLimit = 500

train_features = features [trainTestLimit:]
train_labels = labels [trainTestLimit:]

test_features = features [0:trainTestLimit]
test_labels = labels [0:trainTestLimit]
testData = (test_features, test_labels)


''' build model, add layers '''         
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(11, 21, 18))) # (context, chromatic and enharmonic pitches, other params)
model.add(keras.layers.Dense(45))
model.add(keras.layers.Dense(25, activation=tf.nn.softmax)) # label values       
        


''' compile model '''
model.compile(optimizer="adam",  loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#''' callbacks '''
cb = keras.callbacks.TensorBoard(log_dir='./logs', 
                                 histogram_freq=0, batch_size=1000, write_graph=True, 
                                 write_grads=False, write_images=False, embeddings_freq=0, 
                                 embeddings_layer_names=None, 
                                 embeddings_metadata=None, embeddings_data=None)


''' train model '''
history = model.fit(train_features, train_labels, epochs=200, batch_size=3000, validation_data=testData, callbacks=[cb])
history_dict = history.history


''' evaluate accuracy'''
test_loss, test_acc = model.evaluate(test_features, test_labels)
print('Test accuracy:', test_acc)


''' save model '''
model.save('./model/cadenceModel.h5')


#===============================================================================
# new_model = keras.models.load_model('/Users/Christophe/Desktop/dataset/model.h5')
# new_model.compile(optimizer=tf.train.AdamOptimizer(),  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# new_model.summary()
# 
# loss, acc = new_model.evaluate(test_features, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#===============================================================================


''' graph'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(1, len(acc) + 1)
 
# "bo" is for "blue dot"
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


''' a few predictions '''
#===============================================================================
predictions = model.predict(features)
print ("prediction: " + str(predictions[0]) + "truth: " + str(labels[0]))
print ("prediction: " + str(predictions[1]) + "truth: " + str(labels[1]))
 




