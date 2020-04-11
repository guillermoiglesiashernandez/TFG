import glob, os, json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import seaborn as sns
import zipfile

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from PIL import Image
from sklearn.metrics import plot_confusion_matrix

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class CNN:
    def __init__(self):
        self.lossArray = []
        self.accuracyArray = []
        self.accuracyArrayTest = []
        self.epochLossArray = []
        self.epochAccuracyArray = []
        self.epochAccuracyArrayTest = []

        self.num_imgs = 20000
        self.loadWeights = False

        self.batch_size = 120 #Tiene que ser multiplo de 3
        self.epochs = 20
        self.num_classes = 5
        self.batches = int((self.num_classes*self.num_imgs)/self.batch_size)

        self.input_shape = (128, 128, 1)
        self.savePerformance = 75

        print("-----CREANDO MODELO-----")
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

        if self.loadWeights:
            print("-----CARGANDO PESOS DE LA RED-----")
            self.model.load_weights("saved_model/cnn_weights.hdf5")

            print("-----GUARDANDO BACKUP DEL ENTRENO ANTERIOR-----")
            fileName = "resultsBk" + datetime.now().strftime("%d%m%Y-%H%M%S") + ".zip"
            with zipfile.ZipFile(fileName, "w") as zip:
                zip.write("confMatrix/")
                files = glob.glob("confMatrix/*")
                for file in files:
                    zip.write(file)
                    os.remove(file)

                files = glob.glob("performance/trainPerformance/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/trainPerformance/performance/*.txt")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/trainPerformance/results/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)

                files = glob.glob("performance/testPerformance/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/testPerformance/performance/*.txt")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/testPerformance/results/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)

                files = glob.glob("performance/epochPerformance/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/epochPerformance/performance/*.txt")
                for file in files:
                    zip.write(file)
                    os.remove(file)
                files = glob.glob("performance/epochPerformance/results/*.png")
                for file in files:
                    zip.write(file)
                    os.remove(file)

                for file in glob.glob("saved_model/*"):
                    zip.write(file)

    def save_model(self):
        print("\t Guardando el modelo")
        json_string = self.model.to_json()
        open("saved_model/cnn.json", 'w').write(json_string)
        self.model.save_weights("saved_model/cnn_weights.hdf5")

    def generate_graphic(self):
        print("\t Guardando el grafico de losses")
        x_plot = np.arange(self.lossArray.shape[0])

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(x_plot, self.lossArray)
        ax2.plot(x_plot, self.accuracyArray)
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        filename = 'performance/trainPerformance/results/generated_performance_e%03db%03d.png' % (epoch,batch)
        plt.savefig(filename)
        filename = 'performance/trainPerformance/generated_performance.png'
        plt.savefig(filename)
        plt.close()

        with open('performance/trainPerformance/performance/lossPerformance.txt', 'w') as f:
            for loss in self.lossArray:
                f.write("%s\n" % loss)
        with open('performance/trainPerformance/performance/accuracyPerformance.txt', 'w') as f:
            for accuracy in self.accuracyArray:
                f.write("%s\n" % accuracy)

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

    def saveConfMatrix(self, x_train, y_train, y_trainPredict, epoch, batch):
        print("\t Guardando matriz de confusion")
        confMat = tensorflow.math.confusion_matrix(labels = y_train, predictions = y_trainPredict).numpy()
        confMatNorm = np.around(confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis], decimals=3)

        nImgs = len(y_train)
        accuracyT = (sum(confMat[i][i] for i in range(len(confMat[0]))))/nImgs
        self.accuracyArrayTest = np.append(self.accuracyArrayTest, accuracyT)

        with open('performance/testPerformance/performance/accuracyTestPerformance.txt', 'w') as f:
            for accuracy in self.accuracyArrayTest:
                f.write("%s\n" % accuracy)

        x_plot = np.arange(self.accuracyArrayTest.shape[0])
        plt.tight_layout()
        plt.title('Test accuracy')
        plt.plot(x_plot, self.accuracyArrayTest)

        filename = 'performance/testPerformance/results/generated_accuracy_e%03db%03d.png' % (epoch, batch)
        plt.savefig(filename)
        filename = 'performance/testPerformance/generated_accuracy.png'
        plt.savefig(filename)
        plt.close()

        plt.subplot(211)
        sns.heatmap(confMat, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.suptitle('Numeral confusion matrix')

        plt.subplot(212)
        sns.heatmap(confMatNorm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.suptitle('Percentage confusion matrix')

        filename = 'confMatrix/generated_confMat_e%03db%03d.png' % (epoch,batch)
        plt.savefig(filename)
        plt.close()

    def saveEpochPerformance(self):
        self.epochLossArray = np.append(self.epochLossArray, np.mean(self.lossArray))
        self.epochAccuracyArray = np.append(self.epochAccuracyArray, np.mean(self.accuracyArray))
        self.epochAccuracyArrayTest = np.append(self.epochAccuracyArrayTest, np.mean(self.accuracyArrayTest))

        plt.subplot(311)
        x_plot = np.arange(self.epochLossArray.shape[0])
        plt.tight_layout()
        plt.xlabel('Loss means')
        plt.plot(x_plot, self.epochLossArray)

        plt.subplot(312)
        plt.tight_layout()
        plt.xlabel('Accuracy means')
        plt.plot(x_plot, self.epochAccuracyArray)

        plt.subplot(313)
        plt.tight_layout()
        plt.xlabel('Accuracy tests means')
        plt.plot(x_plot, self.epochAccuracyArrayTest)

        filename = 'performance/epochPerformance/results/generated_performance_e%03d.png' % (epoch)
        plt.savefig(filename)
        filename = 'performance/epochPerformance/generated_performance.png'
        plt.savefig(filename)
        plt.close()

        with open('performance/epochPerformance/performance/lossPerformance.txt', 'w') as f:
            for loss in self.epochLossArray:
                f.write("%s\n" % loss)
        with open('performance/epochPerformance/performance/accuracyPerformance.txt', 'w') as f:
            for accuracy in self.epochAccuracyArray:
                f.write("%s\n" % accuracy)
        with open('performance/epochPerformance/performance/accuracyTestPerformance.txt', 'w') as f:
            for accuracy in self.epochAccuracyArrayTest:
                f.write("%s\n" % accuracy)



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

    for batch in range(cnn.batches - int(cnn.batches*0.2)):
        x_train, y_train = get_batch(datasetX[batch], datasetY[batch], cnn.input_shape)

        loss = cnn.model.train_on_batch(x_train, y_train)
        print(str(batch) + "/" + str(cnn.batches) + "\t[Loss: " + str(loss[0]) +
              "\tAccuracy: " + str(loss[1]) + "]")

        cnn.lossArray = np.append(cnn.lossArray, loss[0])
        cnn.accuracyArray = np.append(cnn.accuracyArray, loss[1])

        if (batch % cnn.savePerformance == 0):
            cnn.save_model()
            cnn.generate_graphic()

            idx = np.random.randint(int(cnn.batches * 0.2), cnn.batches)
            x_predict, y_predict = get_batch(datasetX[idx], datasetY[idx], cnn.input_shape)
            for i in range(10):
                idx = np.random.randint(int(cnn.batches*0.2), cnn.batches)
                x, y = get_batch(datasetX[idx], datasetY[idx], cnn.input_shape)
                x_predict = np.append(x_predict, x, axis=0)
                y_predict = np.append(y_predict, y, axis=0)
            y_prediction = cnn.model.predict_classes(x_predict)
            cnn.saveConfMatrix(x_predict, y_predict, y_prediction, epoch, batch)

    cnn.saveEpochPerformance()