import os
import sys
import psutil
from PIL import Image
import random
import time
import numpy as np
from numba import njit, prange
from moviepy import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

p = psutil.Process()

inputType = sys.argv[1]
loadedImage = sys.argv[2]
outputPath = sys.argv[3]
desiredProcess = sys.argv[4]

width = 64
height = 128

pixel1 = (255,255,255)
pixel2 = (128,128,128)
pixel3 = (0,0,0)
randomInt = random.randint(0,255)
matrix=[]

def loadImage(imagePath):
    global pixelArray
    image = Image.open(imagePath)
    pixelArray = np.array(image)

               

def blur(kernelSize):
    tempArray = np.copy(pixelArray)

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):
            if(rowIndex >= kernelSize and rowIndex <= len(pixelArray) -kernelSize and colIndex >=kernelSize and colIndex <= len(pixelArray[0]) - kernelSize):
                r = []
                g = []
                b = []

                # Loop through neighboring pixels
                for o in range(-kernelSize +1, kernelSize):
                    for p in range(-kernelSize +1, kernelSize):
                        neighbor_pixel = tempArray[rowIndex + o, colIndex + p]
                        r.append(neighbor_pixel[0])
                        g.append(neighbor_pixel[1])
                        b.append(neighbor_pixel[2])

                # Calculate the mean RGB values of the 5x5 neighborhood
                mean_r = int(np.mean(r))
                mean_g = int(np.mean(g))
                mean_b = int(np.mean(b))

                pixelArray[rowIndex, colIndex] = (mean_r, mean_g, mean_b, 255)
                #pixelArray[rowIndex, colIndex] = (r,g,b,255)




#Greatly simplified posterization algorithm
def posterize(numOfColors):
    stepSize = round(256/(numOfColors-1))
    tempArray = np.copy(pixelArray)
    print(stepSize)

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):


            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]
            #print("Before: ",r,", ",g,", ",b)

            #r = np.clip((round(r/ stepSize) * stepSize), 0, 255)
            #g = np.clip((round(g/ stepSize) * stepSize), 0, 255)
            b = np.clip((round(b/ stepSize) * stepSize), 0, 255)
            #print("After: ",r,", ",g,", ",b)

            pixelArray[rowIndex, colIndex] = (b, b, b, 255)    


def ditherVideo(videoPath):
    imageArray = []
    frameDirectory = "frames/"
    clip = VideoFileClip(videoPath)
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype='uint8')):
        framePath = f"{frameDirectory}frame_{i:04d}.png"
        global pixelArray
        pixelArray = np.dstack([frame, np.full(frame.shape[:2], 255, dtype="uint8")])  # Convert to RGBA
        posterizeDither(2)
        imageArray.append(pixelArray)

    clip = ImageSequenceClip(imageArray, fps=clip.fps)
    clip.write_videofile(outputPath, codec="libx264")
    print(f"Video saved to {outputPath}")
    



#Ordered dithering
#@njit(parallel=True)
def posterizeDither(numOfColors):
    stepSize = round(256/(numOfColors-1))
    tempArray = np.copy(pixelArray)
    rule = []

    matrixSize = 2
    
    if(matrixSize == 2):
        rule = stepSize * (1.0 / 4.0) * np.array([
            [1, 3],
            [3.5, 2]
        ])

    if(matrixSize == 4):
        rule = stepSize * (1.0 / 16.0) * np.array([
            [1, 9, 3, 11],
            [13, 5, 15, 7],
            [4, 12, 2, 10],
            [16, 8, 14, 6]
        ])

    if(matrixSize == 8):
        rule = stepSize * (1.0 / 64.0) * np.array([
            [1, 49, 13, 61, 4, 52, 16, 63],
            [33, 17, 45, 29, 36, 20, 48, 32],
            [9, 57, 5, 53, 12, 60, 8, 56],
            [41, 25, 37, 21, 44, 28, 40, 24],
            [3, 51, 15, 63, 2, 50, 14, 62],
            [35, 19, 47, 31, 34, 18, 46, 30],
            [11, 59, 7, 55, 10, 58, 6, 54],
            [43, 27, 39, 23, 42, 26, 38, 22]
        ])


    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):



            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]
            a = tempArray[rowIndex, colIndex][3]

            ditherThreshold = rule[rowIndex % matrixSize, colIndex % matrixSize]-1
            #print("DIther threshold: ",ditherThreshold)
            
            newR = clip(dither(r, stepSize, ditherThreshold), 0, 255)
            newG = clip(dither(g, stepSize, ditherThreshold), 0, 255)
            newB = clip(dither(b, stepSize, ditherThreshold), 0, 255)
            newA = clip(dither(a, stepSize, ditherThreshold), 0, 255)
            pixelArray[rowIndex, colIndex] = (newR, newG, newB, newA)   
    print("Dither applied")

@njit
def dither(color,stepSize, ditherThreshold):
    #quantizedColor = round(round(color/ stepSize) * stepSize)
    quantizedColor = 0
    #print("quantizedCOlor: ",quantizedColor)
    #print("Color",color)
    if(color % stepSize) >= ditherThreshold:
        quantizedColor = clip(quantizedColor + stepSize, 0, 255)
        #quantizedColor = 255
        #print("True")
    else:
        pass
    return quantizedColor

@njit
def clip(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


def saveImage(imageName):
    img = Image.fromarray(pixelArray, 'RGBA')
    img = img.convert("P", palette=Image.ADAPTIVE, colors=8)
    # Save the image
    img.save(imageName, optimize=True)
    print(f"Image saved as ",imageName)



def generateRandomNoise():
    matrix.clear()
    for row in range(height):
        a=[]
        for column in range(width):
            randomInt = random.randint(1,3)
            #for p in range(16):
            if(randomInt == 1):
                a.append(pixel1)
            elif(randomInt==2):
                a.append(pixel2)
            elif(randomInt==3):
                a.append(pixel3)

        matrix.append(a)


if inputType == "image":
    loadImage(loadedImage)
    if desiredProcess == "dither":
        posterizeDither(2)
        saveImage(outputPath)
    elif desiredProcess == "posterize":
        posterize(8)
        saveImage(outputPath)
    else:
        print("Please enter either 'dither' or 'posterize'")
elif inputType == "video":
    if desiredProcess == "dither":
        ditherVideo(loadedImage)
    else:
        print("Only video dithering is currently supported")

print("Peak memory usage: ",(p.memory_info().peak_wset/1000)/1000,"MB")

#loadImage("images.jpg")
#modifyImage()
#blur(4)
#posterize(5)
#posterize2(2)
#ditherVideo("videos/blizzard-small.mp4")
#assemble_video_from_frames("frames/", "videos/output.mp4", fps=24)
#posterizeDither(2)
#saveModifiedImage("new-image.png")
#saveImage()
