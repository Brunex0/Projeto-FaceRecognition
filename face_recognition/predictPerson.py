from glob import glob
from loadContentFiles import load_yaml
from auxFunctions.auxiliarForLFW import *
from auxFunctions.auxiliarForICBRW import *
from model import ResNet50WithSoftmax
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
    #create labels
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


    idForPhotos=3

    # Save all the identities
    # saveIdentities(cfgData,model_part, 90,idForPhotos)

    distances = []

    photo = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_ProbeImages_mtcnn_224\\3\\004_04.png"
    photoID=3
    vector = nearestFace(prevision(cfgData, model_part, photo))

    vector1 = np.argsort(vector)[::-1]

    from colorama import Fore, Back, Style
    print("ID original do indivíduo:", photoID)
    for i in range(0,10):
        print(Fore.CYAN + "ID previsto:", vector1[i], "Rank:", i+1)
        if vector1[i] == photoID:
            break




