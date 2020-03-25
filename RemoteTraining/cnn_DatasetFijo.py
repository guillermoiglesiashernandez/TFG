import glob, os, json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from PIL import Image
from sklearn.metrics import plot_confusion_matrix


class CNN:
    def __init__(self):
        self.lossArray = []
        self.accuracyArray = []

        self.num_imgs = 10000

        self.batch_size = 60 #Tiene que ser multiplo de 3
        self.epochs = 5000
        self.num_classes = 3
        self.batches = int((self.num_classes*self.num_imgs)/self.batch_size)

        self.input_shape = (128, 128, 1)
        self.savePerformance = 20

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=self.input_shape))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='linear', activity_regularizer=tensorflow.keras.regularizers.l1(0.0001)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='linear', activity_regularizer=tensorflow.keras.regularizers.l1(0.0001)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.add(Dropout(0.2))
        self.model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])
        print(self.model.summary())

    def save_model(self):
        print("\t Guardando el modelo")

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.model, "cnn")

    def generate_graphic(self):
        print("\t Guardando el grafico de losses")
        x_plot = np.arange(self.lossArray.shape[0])

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(x_plot, self.lossArray)
        ax2.plot(x_plot, self.accuracyArray)
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        filename = 'graphics/generated_graphic_e%03d.png' % (epoch)
        plt.savefig(filename)
        plt.close()

    def savePrediction(self, x_train, y_train, y_trainPredict):
        print("\t Guardando predicciones")
        x_train = x_train.reshape(1,512,512)
        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)

        a.set_title(str(y_train) + "/" + str(y_trainPredict))
        plt.imshow(x_train[0], cmap="gray")
        filename = 'images/generated_prediction_e%03d.png' % (epoch)
        plt.savefig(filename)
        plt.close()

    def saveConfMatrix(self, x_train, y_train, y_trainPredict):
        print("\t Guardando matriz de confusion")
        confMat = tensorflow.math.confusion_matrix(labels = y_train, predictions = y_trainPredict).numpy()

        figure = plt.figure()
        sns.heatmap(confMat, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        filename = 'confMatrix/generated_confMat_e%03d.png' % (epoch)
        plt.savefig(filename)
        plt.close()

def get_imgs(n_imgs):
    print("-----CARGANDO DATASET-----")

    CLASS0 = 0
    CLASS1 = 0
    CLASS2 = 0
    x = [[],[],[]]
    y = [[],[],[]]
    i = 0
    j = 0

    while (i<16 and (CLASS0<n_imgs or CLASS1<n_imgs or CLASS2<n_imgs)):
        with open(('C:/Users/Guillermo/Desktop/TFG/DatasetBW/MetadataImg/' + str(i) + '.json'), 'r', encoding="utf8") as f:
            data = json.load(f)
        print('C:/Users/Guillermo/Desktop/TFG/DatasetBW/MetadataImg/' + str(i) + '.json')

        j=0
        while(j<len(data) and (CLASS0<n_imgs or CLASS1<n_imgs or CLASS2<n_imgs)):
            fold = '%04d' % (int(data[j]["id"]) % 1000)
            try:
                imgFile = "C:/Users/Guillermo/Desktop/TFG/DatasetBW/LowResImages/" + fold + "/" + str(data[j]["id"]) + ".jpg"
                img = plt.imread(imgFile)

                if (data[j]["tagArray"][11] == 1 and CLASS0<n_imgs):
                    # Sombrero
                    x[0].append(imgFile)
                    y[0].append("0")
                    CLASS0 += 1
                elif (data[j]["tagArray"][0] == 1 and data[j]["tagArray"][3] == 1 and CLASS1<n_imgs):
                    # Chica con el pelo corto
                    x[1].append(imgFile)
                    y[1].append("1")
                    CLASS1 += 1
                elif (data[j]["tagArray"][0] == 1 and data[j]["tagArray"][1] == 1 and CLASS2<n_imgs):
                    # Chica con el pelo largo
                    x[2].append(imgFile)
                    y[2].append("2")
                    CLASS2 += 1
            except FileNotFoundError:
                ""
            j+=1
        i+=1

    print("Imagenes cargadas:\n"
          "\tClase 0: " + str(CLASS0) + "\tClase 1: " + str(CLASS1) + "\tClase 2: " + str(CLASS2))
    return x, y

def mix_dataset(x_dataset, y_dataset, n_batches, n_imgs):
    x_train = [[None for m in range(n_imgs)] for n in range(n_batches)]
    y_train = [[None for m in range(n_imgs)] for n in range(n_batches)]

    aux = 0
    for i in range(n_batches):
        j = 0
        while j<(n_imgs):
            x_train[i][j] = x_dataset[0][aux]
            y_train[i][j] = y_dataset[0][aux]
            j += 1
            x_train[i][j] = x_dataset[1][aux]
            y_train[i][j] = y_dataset[1][aux]
            j += 1
            x_train[i][j] = x_dataset[2][aux]
            y_train[i][j] = y_dataset[2][aux]
            j += 1
            aux += 1

    return x_train, y_train

def load_dataset(n_imgs, batch_size):
    x_dataset, y_dataset = get_imgs(n_imgs)
    n_batches = int((n_imgs*3)/batch_size)
    x_train, y_train = mix_dataset(x_dataset, y_dataset, n_batches, batch_size)

    return x_train, y_train

def get_batch(x_dataset, y_dataset, input_shape):
    x_train = [None for m in range(len(x_dataset))]
    y_train = [None for m in range(len(y_dataset))]

    y_train = y_dataset
    for it in range(len(x_dataset)):
        x_train[it] = plt.imread(x_dataset[it])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])

    x_train = x_train.astype('float32')
    y_train = y_train.astype('int32')
    np.true_divide(x_train, 255)

    return x_train, y_train

cnn = CNN()

datasetX, datasetY = load_dataset(cnn.num_imgs, cnn.batch_size) #aqui no va 300, va cnn.batch_size
for epoch in range(cnn.epochs):
    print("-----EPOCH " + str(epoch) +"/" + str(cnn.epochs) + "-----")

    for batch in range(cnn.batches):
        x_train, y_train = get_batch(datasetX[epoch], datasetY[epoch], cnn.input_shape)

        loss = cnn.model.train_on_batch(x_train, y_train)
        print(str(batch) + "/" + str(cnn.batches) + "\t[Loss: " + str(loss[0]) +
              "\tAccuracy: " + str(loss[1]) + "]")

        cnn.lossArray = np.append(cnn.lossArray, loss[0])
        cnn.accuracyArray = np.append(cnn.accuracyArray, loss[1])

        if (batch % cnn.savePerformance == 0):
            cnn.save_model()
            cnn.generate_graphic()
            x_train, y_train = get_batch(datasetX[0], datasetY[0], cnn.input_shape)
            y_trainPredict = cnn.model.predict_classes(x_train)
            cnn.saveConfMatrix(x_train, y_train, y_trainPredict)
