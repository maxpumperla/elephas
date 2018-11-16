from elephas.java import java_classes, adapter
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.save('test.h5')


kmi = java_classes.KerasModelImport
file = java_classes.File("test.h5")

java_model = kmi.importKerasSequentialModelAndWeights(file.absolutePath)

weights = adapter.retrieve_keras_weights(java_model)
model.set_weights(weights)