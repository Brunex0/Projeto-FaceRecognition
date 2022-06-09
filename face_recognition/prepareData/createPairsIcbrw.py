import itertools
import glob
import random
from loadContentFiles import load_yaml

cfgData = load_yaml('../config.yml')
pairsFile = open(cfgData['pairs-local'], 'a')
PATH = cfgData['dataset-local']

# for the true pairs
for i in range(0, 90):
    images = glob.glob(PATH + str(i) + '/*')
    combinations = itertools.combinations(images, 2)
    for item in combinations:
        # 1 means that they are the same person
        pairsFile.write(item[0] + '\t' + item[1] + '\t' + '1' + '\n')

# For the false pairs 0 = false pair
for i in range(0, 90):
    imagesFromSamePerson = glob.glob(PATH + str(i) + '/*')
    imagesFromOtherIdentities = glob.glob(PATH + '[!' + str(i) + ']*/*')
    for image in imagesFromSamePerson:
        # Pick to random images from the other identities
        randImage1 = random.choice(imagesFromOtherIdentities)
        randImage2 = random.choice(imagesFromOtherIdentities)

        while randImage1 == randImage2:
            randImage1 = random.choice(imagesFromOtherIdentities)
            randImage2 = random.choice(imagesFromOtherIdentities)
        # 0 means that they aren't the same person
        pairsFile.write(image + '\t' + randImage1 + '\t' + '0' + '\n')
        pairsFile.write(image + '\t' + randImage2 + '\t' + '0' + '\n')

pairsFile.close()
