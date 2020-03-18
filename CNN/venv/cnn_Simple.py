import glob, os, json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from PIL import Image

batch_size = 1
num_classes = 7
batches = 100

img_rows, img_cols = 512, 512

def load_data():
    print("---LOADING DATA---")
    trainX = []
    trainY = []
    testX = []
    testY = []
    cont1 = 0
    cont2 = 0
    cont3 = 0
    cont4 = 0
    cont5 = 0
    cont6 = 0
    cont7 = 0
    cont8 = 0
    cont9 = 0
    cont10 = 0
    cont11 = 0
    cont12 = 0
    img = None

    for folder in glob.glob('C:/Users/Guillermo/Desktop/Metadata/*'):
        print(folder)
        with open(folder, 'r', encoding="utf8") as f:
            data = json.load(f)
        for element in data:
            fold = '%04d' % (int(element["id"]) % 1000)
            try:
                img = plt.imread("C:/Users/Guillermo/Desktop/TFG_Dataset/venv/BWImages/"
                               + fold + "/" + str(element["id"]) + ".jpg")
                folderName = os.path.basename(folder)
                folderName = os.path.splitext(folderName)[0]
                if (int(folderName) <= 0):
                    if (element["tagArray"][11] == 1):
                        # Sombrero
                        trainX.append(img)
                        trainY.append("6")
                        cont7 += 1
                    elif (element["tagArray"][10] == 1):
                        # Falda
                        trainX.append(img)
                        trainY.append("5")
                        cont6 += 1
                    elif (element["tagArray"][6] == 1):
                        # Rubia
                        trainX.append(img)
                        trainY.append("4")
                        cont5 += 1
                    elif (element["tagArray"][5] == 1):
                        # Sonrojada
                        trainX.append(img)
                        trainY.append("3")
                        cont4 += 1
                    elif (element["tagArray"][0] == 1 and element["tagArray"][3] == 1):
                        # Chica con el pelo corto
                        trainX.append(img)
                        trainY.append("2")
                        cont3 += 1
                    elif (element["tagArray"][0] == 1 and element["tagArray"][2] == 1):
                        # Chica sonriendo
                        trainX.append(img)
                        trainY.append("1")
                        cont2 += 1
                    elif(element["tagArray"][0] == 1 and element["tagArray"][1] == 1):
                        # Chica con el pelo largo
                        trainX.append(img)
                        trainY.append("0")
                        cont1 += 1
                else:
                    if (element["tagArray"][11] == 1):
                        # Sombrero
                        testX.append(img)
                        testY.append("6")
                        cont7 += 1
                    elif (element["tagArray"][10] == 1):
                        # Falda
                        testX.append(img)
                        testY.append("5")
                        cont6 += 1
                    elif (element["tagArray"][6] == 1):
                        # Rubia
                        testX.append(img)
                        testY.append("4")
                        cont5 += 1
                    elif (element["tagArray"][5] == 1):
                        # Sonrojada
                        testX.append(img)
                        testY.append("3")
                        cont4 += 1
                    elif (element["tagArray"][0] == 1 and element["tagArray"][3] == 1):
                        # Chica con el pelo corto
                        testX.append(img)
                        testY.append("2")
                        cont3 += 1
                    elif (element["tagArray"][0] == 1 and element["tagArray"][2] == 1):
                        # Chica sonriendo
                        testX.append(img)
                        testY.append("1")
                        cont2 += 1
                    if (element["tagArray"][0] == 1 and element["tagArray"][1] == 1):
                        # Chica con el pelo largo
                        testX.append(img)
                        testY.append("0")
                        cont1 += 1
            except FileNotFoundError:
                ""

    print("---DATA LOADED---")
    print("\nN IMGS DE CATEGORIAS\n")
    print("Categoria 0: " + str(cont1))
    print("Categoria 1: " + str(cont2))
    print("Categoria 2: " + str(cont3))
    print("Categoria 3: " + str(cont4))
    print("Categoria 4: " + str(cont5))
    print("Categoria 5: " + str(cont6))
    print("Categoria 6: " + str(cont7))
    return (trainX, trainY), (testX, testY)

(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

if K.image_data_format() == 'channels_first':
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train[:1000,:,:,:]
y_train = y_train[:1000]
x_test = x_test[:1000,:,:,:]
y_test = y_test[:1000]

x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
y_train = y_train.astype('int16')
y_test = y_test.astype('int16')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_train.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

for batch in range(batches):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    traX = x_train[idx]
    traY = y_train[idx]
    loss = model.train_on_batch(traX, traY)
    print(str(batch) + "/" + str(batches) + " [Loss: " + str(loss) + "]")

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])