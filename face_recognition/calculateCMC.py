import os.path

import tensorflow as tf
import numpy as np
from model import ResNet50WithSoftmax
from glob import glob
from loadContentFiles import load_yaml
from auxFunctions.auxiliarForLFW import *
from auxFunctions.auxiliarForICBRW import *
from auxFunctions.auxiliarForROC import *
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

import seaborn as sns


def prevision(cfgData, model, url):
    """
    Do the inference of a image and return the embeddings/features of that image
    :param cfgData: the config data
    :param model: the CNN model
    :param url: the image to input to the CNN
    :return: the vector of features
    """
    img = cv2.imread(url)
    img = cv2.resize(img, (cfgData['inputSize'], cfgData['inputSize']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds1 = model(img)
    if cfgData['test-type'] == 'Align-L2' or cfgData['test-type'] == 'Align-L2-CosineSim':
        embeds1 = K.l2_normalize(embeds1, axis=1)

    return embeds1


def saveIdentities(cfgData, model, n_classes,id):
    """
    Save the identities (feature vector) of each person that is in the dataset
    :param cfgData: the config file
    :param model: the model
    :param n_classes: the numer of identities
    """
    # create labels
    labels = []
    labels2 = []
    Identities = []
    for i in range(1, n_classes + 1):
        labels.append(str(i).zfill(3))
        labels2.append(str(i - 1))

    for i in range(0, n_classes):
        url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_ProbeImages_mtcnn_224\\" + labels2[
            i] + "\\" + labels[i] + "_0"+str(id)+".png"
        # url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_GalleryImages_mtcnn_224\\" + labels2[i] + "\\" + labels[i] + "_f.png"
        print(url)
        X = prevision(cfgData, model, url)
        Identities.append(X.numpy())

    data = np.asarray(Identities)
    np.save('Identities.npy', data)


def nearestFace(X):
    """
    Determines how far away a feature vector is from each of the identities, and save it in a vector
    :param X: the person's features
    :return: the distance array with all the distances
    """
    dist = []
    data = np.load('Identities.npy')
    for i in range(0, 90):
        tensor = tf.convert_to_tensor(data[i], dtype=tf.float32)
        dist.append(metrics.pairwise.cosine_similarity(tensor, X)[0][0])
        # dist.append(-metrics.pairwise.euclidean_distances(tensor, X)[0][0])
    return dist


if "__main__":
    cfgData = load_yaml('config.yml')
    folders = glob(cfgData['train-path'] + "/*")
    modelTest = ResNet50WithSoftmax(cfgData, 10575)
    modelTest.load_weights(cfgData['model-weights-path'])

    model_part = tf.keras.Model(
        inputs=modelTest.input,
        outputs=modelTest.get_layer("flatten").output
    )

    # Save all the identities
    #saveIdentities(cfgData,model_part, 90)
    #distances = []
    graphsBYID=[]
    idForPhotos=3
    # Determine the array of distances for all the images in the dataset
    for id in range(1,6):
        idToCancel = id
        saveIdentities(cfgData, model_part, 90, id)
        distances = []
        for i in range(0, 90):
            for j in range(1, 6):
                if idToCancel != j:
                    url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_ProbeImages_mtcnn_224\\" + str(
                        i) + "\\" + str(i + 1).zfill(3) + "_" + str(j).zfill(2) + ".png"
                    print(url)
                    # X = prevision(cfgData, model, url)
                    vector = nearestFace(prevision(cfgData, model_part, url))

                    vector1 = np.argsort(vector)[::-1]

                    distances.append(vector1)

        matrix = []
        print(len(distances))

        # locates where the person is in the array and marks that position as 1 and the rest as 0
        id2 = 0
        count = 0
        distancesIDLocation = []
        for value in distances:
            idLocation = np.where(value == id2)[0][0]
            array = np.zeros(90, dtype=int)
            array[idLocation] = 1
            # array com o idlocation 10 piores ou 25 piores
            distancesIDLocation.append(idLocation)
            matrix.append(array)
            count += 1
            if count % 4 == 0:
                id2 += 1

        #Get the location of the 10 worst photos
        if(idForPhotos == id):
            #Get the index of the 10 max values
            worst = np.argpartition(distancesIDLocation, -10)[-10:]

        # Determine the values for each rank
        rank = []
        rank.append(np.array(matrix)[:, 0])
        for i in range(1, 90):
            rank.append(np.array(matrix)[:, i])
            rank[i] = rank[i] + rank[i - 1]
        import matplotlib.pyplot as plt

        graph = []
        for item in rank:
            graph.append(sum(item) / 360)
        graphsBYID.append(graph)


    graph=np.mean(graphsBYID, axis=0)


    maxPlot=[]
    minPlot=[]
    for i in range(0,90):
        maxi = max([graphsBYID[0][i],graphsBYID[1][i],graphsBYID[2][i],graphsBYID[3][i],graphsBYID[4][i]])
        mini = min([graphsBYID[0][i],graphsBYID[1][i],graphsBYID[2][i],graphsBYID[3][i],graphsBYID[4][i]])
        maxPlot.append(maxi)
        minPlot.append(mini)

    # Create the CMC curve
    sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
    sns.color_palette('pastel')
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)  # legend fontsize
    plt.rc('font', size=13)  # controls default text sizes
    plt.figure(figsize=(7, 7), tight_layout=True)
    plt.plot(range(1, len(graph) + 1), graph, 'r')
    plt.plot(range(1, len(graph) + 1), maxPlot, 'b', alpha=.4)
    plt.plot(range(1, len(graph) + 1), minPlot, 'b',alpha=.4)
    plt.fill_between(range(1, len(graph) + 1),maxPlot,minPlot, alpha=.3)
    plt.xlabel('Rank')
    plt.ylabel('Identification Accuracy')
    plt.title('CMC')
    plt.legend(title='CMC', title_fontsize=13, labels=['Mean'], loc='lower right')
    plt.savefig('E:\\Projeto-FaceRecognition\\face_recognition\\graphs\\CMCCurve3.png')



    # Get the worst photos to identify
    print(worst)
    for element in worst:
        print("folder:", element // 4, "photo:", (element % 4))

    for i in range(0, 10):
        print(i, ":", graph[i])
