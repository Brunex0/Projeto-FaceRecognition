import os

import numpy as np
from PIL import Image
import cv2



def CropImage(Info, directoryPath, destinationPath):
    """
        Crop the image and save it in the correct folder. Each identity as a folder
    :param Info: the information about all the images of the datastet
    :param directoryPath: directory where the images are
    :param destinationPath: directory were the images are stored
    """

    # Go through the list and crop each image.
    for photo in Info:
        photoName = photo.split(" ")[0]
        X = int(photo.split(" ")[1])
        Y = int(photo.split(" ")[2])
        width = int(photo.split(" ")[3])
        height = int(photo.split(" ")[4])

        image = Image.open(directoryPath + '\\' + photoName)

        image_Cropped = image.crop((X, Y, X + width, Y + height))

        if not os.path.exists(destinationPath + '\\' + str(int(photoName.split('_')[0])-1)):
            os.makedirs(destinationPath + '\\' + str(int(photoName.split('_')[0])-1))
        # Save the image in the destination path
        """if ('_05' in photoName or '_04' in photoName or '_03' in photoName) and ('Probe' in destinationPath):
            image_Cropped.save('..\\Dataset\\icbrw_Data_Cropped\\icbrw_GalleryImages' + '\\' + str(int(photoName.split('_')[0])-1) + '\\' + photoName)
        else:
            image_Cropped.save(destinationPath + '\\' + str(int(photoName.split('_')[0])-1) + '\\' + photoName)"""

        image_Cropped.save(destinationPath + '\\' + photoName)

def main():

    # Set the path to the gallery Images and ProbelImages
    ProbefinalPath = '..\\Dataset\\icbrw_Data\\annotations_ProbeImages.txt'
    GalleryfinalPath = '..\\Dataset\\icbrw_Data\\annotations_GalleryImages.txt'

    # Open the files
    galleryFile = open(GalleryfinalPath)
    probeFile = open(ProbefinalPath)

    # Read the lines of the files and save the data in a list
    galleryInfo = galleryFile.readlines()
    probeInfo = probeFile.readlines()

    # Set the inicialPath of the images and the destination path of the cropped images (GalleryImages)
    inicialPath = '..\\Dataset\\icbrw_Data\\icbrw_GalleryImages'
    destinationPath = '..\\Dataset\\icbrw_Data_Cropped\\icbrw_GalleryImages'
    CropImage(galleryInfo, inicialPath, destinationPath)

    # Set the inicialPath of the images and the destination path of the cropped images (ProbeImages)
    inicialPath = '..\\Dataset\\icbrw_Data\\icbrw_ProbeImages'
    destinationPath = '..\\Dataset\\icbrw_Data_Cropped\\icbrw_ProbeImages'
    CropImage(probeInfo, inicialPath, destinationPath)


main()
