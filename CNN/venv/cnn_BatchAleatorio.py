import glob, os, json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from PIL import Image
from sklearn.metrics import plot_confusion_matrix

CLASS1=0
CLASS2=0
CLASS3=0
DIFF_IMG=10

class CNN:
    def __init__(self):
        self.lossArray = []
        self.accuracyArray = []

        self.batch_size = 32
        self.num_classes = 3
        self.batches = 10
        self.epochs = 100
        self.input_shape = (512, 512, 1)

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
        self.model.add(Dense(256, activation='relu'))
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

def getTag(y):
    i = 0
    y_predict = -1
    while (y_predict == -1):
        if(y[i] == 1):
            y_predict = i

def load_data(batch_size, countImg=False):
    global CLASS1, CLASS2, CLASS3
    x = []
    y = []

    idx1 = np.random.randint(0, len(glob.glob('C:/Users/Guillermo/Desktop/TFG/DatasetBW/MetadataImg/*')))
    with open(('C:/Users/Guillermo/Desktop/TFG/DatasetBW/MetadataImg/' + str(idx1) + '.json'), 'r', encoding="utf8") as f:
        data = json.load(f)

    cont = 0
    while(cont<batch_size):
        idx2 = np.random.randint(0, len(data))
        fold = '%04d' % (int(data[idx2]["id"]) % 1000)
        try:
            img = plt.imread("C:/Users/Guillermo/Desktop/TFG/GenerateBWDataset/venv/BWImages/"
                             + fold + "/" + str(data[idx2]["id"]) + ".jpg")

            if (data[idx2]["tagArray"][11] == 1):
                # Sombrero
                if ((CLASS1 < CLASS2 + DIFF_IMG and CLASS1 < CLASS3 + DIFF_IMG) or countImg == False):
                    x.append(img)
                    y.append("0")
                    cont += 1

                    if(countImg==True):
                        CLASS1+=1
            elif (data[idx2]["tagArray"][0] == 1 and data[idx2]["tagArray"][3] == 1):
                # Chica con el pelo corto
                if ((CLASS2 < CLASS1 + DIFF_IMG and CLASS2 < CLASS3 + DIFF_IMG) or countImg == False):
                    x.append(img)
                    y.append("1")
                    cont += 1

                    if (countImg == True):
                        CLASS2+=1
            elif (data[idx2]["tagArray"][0] == 1 and data[idx2]["tagArray"][1] == 1):
                # Chica con el pelo largo
                if((CLASS3 < CLASS1 + DIFF_IMG and CLASS3 < CLASS2 + DIFF_IMG) or countImg == False):
                    x.append(img)
                    y.append("2")
                    cont += 1

                    if (countImg == True):
                        CLASS3+=1
        except FileNotFoundError:
            ""

    return x, y

def get_imgs(batch_size, input_shape, countImg=False):
    x_train, y_train = load_data(batch_size, countImg=countImg)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])

    x_train = x_train.astype('float32')
    y_train = y_train.astype('int32')
    np.true_divide(x_train, 255)
    #x_train /= 255

    return x_train, y_train

cnn = CNN()

for epoch in range(cnn.epochs):
    print("-----EPOCH " + str(epoch) +"/" + str(cnn.epochs) + "-----")
    for batch in range(cnn.batches):
        x_train, y_train = get_imgs(cnn.batch_size, cnn.input_shape, countImg=True)
        loss = cnn.model.train_on_batch(x_train, y_train)
        print(str(batch) + "/" + str(cnn.batches) + "\t[Loss: " + str(loss[0]) +
              "\tAccuracy: " + str(loss[1]) + "]")

        if(batch%1 == 0):
            cnn.lossArray = np.append(cnn.lossArray, loss[0])
            cnn.accuracyArray = np.append(cnn.accuracyArray, loss[1])

    #x_train, y_train = get_imgs(1, cnn.input_shape)
    #y_trainPredict = cnn.model.predict(x_train)
    #cnn.savePrediction(x_train, y_train, y_trainPredict)
    cnn.save_model()
    cnn.generate_graphic()
    x_train, y_train = get_imgs(100, cnn.input_shape)
    y_trainPredict = cnn.model.predict_classes(x_train)
    cnn.saveConfMatrix(x_train, y_train, y_trainPredict)

    print("NºImagenes de cada clase:\n"
          "\tClase 1: " + str(CLASS1) +
          "\tClase 2: " + str(CLASS2) +
          "\tClase 3: " + str(CLASS3))
