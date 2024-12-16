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

def loadImage(imagePath):
    global pixelArray
    image = Image.open(imagePath)
    pixelArray = np.array(image)

               
@njit
def blur(kernelSize, inputArray):
    outputArray = np.copy(inputArray)

    for rowIndex, pixelColumn in enumerate(inputArray):
        for colIndex, pixel in enumerate(pixelColumn):
            if(rowIndex >= kernelSize and rowIndex <= len(inputArray) -kernelSize and colIndex >=kernelSize and colIndex <= len(inputArray[0]) - kernelSize):
                r = []
                g = []
                b = []

                # Loop through neighboring pixels
                for o in range(-kernelSize +1, kernelSize):
                    for p in range(-kernelSize +1, kernelSize):
                        neighbor_pixel = outputArray[rowIndex + o, colIndex + p]
                        r.append(neighbor_pixel[0])
                        g.append(neighbor_pixel[1])
                        b.append(neighbor_pixel[2])

                # Calculate the mean RGB values current pixel and its neighbours
                meanR = int(np.mean(np.asarray(r)))
                meanG = int(np.mean(np.asarray(g)))
                meanB = int(np.mean(np.asarray(b)))

                outputArray[rowIndex, colIndex] = (meanR, meanG, meanB, 255)
    return outputArray


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

            r = np.clip((round(r/ stepSize) * stepSize), 0, 255)
            g = np.clip((round(g/ stepSize) * stepSize), 0, 255)
            b = np.clip((round(b/ stepSize) * stepSize), 0, 255)
            #print("After: ",r,", ",g,", ",b)

            pixelArray[rowIndex, colIndex] = (r, g, b, 255)    


def ditherVideo(videoPath):
    imageArray = []
    clip = VideoFileClip(videoPath)
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype='uint8')):
        global pixelArray
        pixelArray = np.dstack([frame, np.full(frame.shape[:2], 255, dtype="uint8")])  # Convert to RGBA
        pixelArray = posterizeDither(2, pixelArray)
        imageArray.append(pixelArray)

    clip = ImageSequenceClip(imageArray, fps=clip.fps)
    clip.write_videofile(outputPath, codec="libx264")
    print(f"Video saved to {outputPath}")
    
def ditherGif(videoPath):
    imageArray = []
    clip = VideoFileClip(videoPath)
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype='uint8')):
        global pixelArray
        pixelArray = np.dstack([frame, np.full(frame.shape[:2], 255, dtype="uint8")])  # Convert to RGBA
        pixelArray = posterizeDither(2, pixelArray)
        imageArray.append(pixelArray)

    clip = ImageSequenceClip(imageArray, fps=clip.fps)
    clip.write_gif(outputPath)
    print(f"Video GIF to {outputPath}")


#Ordered dithering
@njit(parallel=True)
def posterizeDither(numOfColors, inputPixels):
    stepSize = round(256/(numOfColors-1))
    outputPixels = np.copy(inputPixels)


    matrixSize = 8
    #2x2 matrix
    if(matrixSize == 2):
        rule = stepSize * (1.0 / 4.0) * (np.array([
            [1, 3],
            [4, 2]
        ]) - 0.5)
    #4x4 matrix
    elif(matrixSize == 4):
        rule = stepSize * (1.0 / 16.0) * (np.array([
            [1, 9, 3, 11],
            [13, 5, 15, 7],
            [4, 12, 2, 10],
            [16, 8, 14, 6]
        ]) - 0.5)
    #8x8 matrix
    else:
        rule = stepSize * (1.0 / 64.0) * (np.array([
            [1, 37, 9, 45, 3, 39, 11, 47],
            [49, 17, 57, 25, 51, 19, 59, 27],
            [13, 41, 5, 33, 15, 43, 7, 35],
            [61, 29, 53, 21, 63, 31, 55, 23],
            [4, 40, 12, 48, 2, 38, 10, 46],
            [52, 20, 60, 28, 50, 18, 58, 26],
            [16, 44, 8, 36, 14, 42, 6, 34],
            [64, 32, 56, 24, 62, 30, 54, 22]
        ]) - 0.5)


    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):

            r = inputPixels[rowIndex, colIndex][0]
            g = inputPixels[rowIndex, colIndex][1]
            b = inputPixels[rowIndex, colIndex][2]
            a = inputPixels[rowIndex, colIndex][3]

            ditherThreshold = rule[rowIndex % matrixSize, colIndex % matrixSize]-1
            #ditherThreshold = (stepSize * (1.0 / 64.0) * (random.randint(1, 64) -0.5)) - 1
            #print("DIther threshold: ",ditherThreshold)
            
            newR = clip(dither(r, stepSize, ditherThreshold), 0, 255)
            newG = clip(dither(g, stepSize, ditherThreshold), 0, 255)
            newB = clip(dither(b, stepSize, ditherThreshold), 0, 255)
            newA = clip(dither(a, stepSize, ditherThreshold), 0, 255)
            outputPixels[rowIndex, colIndex] = (newR, newG, newB, newA)   
    print("Dither applied")
    return outputPixels


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


@njit
def edgeDetection(edgeThreshold, inputPixels):
    outputPixels = np.copy(inputPixels)
    edgeColor = (0,0,0, 255)
    backgroundColor = (255,255,255, 255)
    shadeColor = (128,128,128, 255)

    for rowIndex, pixelColumn in enumerate(inputPixels):
        for colIndex, pixel in enumerate(pixelColumn):
            r = inputPixels[rowIndex, colIndex][0]
            g = inputPixels[rowIndex, colIndex][1]
            b = inputPixels[rowIndex, colIndex][2]
            blackAndWhite = int(0.30*r + 0.59*g + 0.11*b)
            
            inputPixels[rowIndex, colIndex] = (blackAndWhite,blackAndWhite,blackAndWhite, 255)

    for rowIndex, pixelColumn in enumerate(inputPixels):
        for colIndex, pixel in enumerate(pixelColumn):

            if(rowIndex >= 2 and rowIndex <= len(inputPixels) -2 and colIndex >=2 and colIndex <= len(inputPixels[0]) - 2):
                if(inputPixels[rowIndex, colIndex][0] > inputPixels[rowIndex+1, colIndex][0]*edgeThreshold) or (inputPixels[rowIndex, colIndex][0] > inputPixels[rowIndex, colIndex+1][0]*edgeThreshold) or (inputPixels[rowIndex, colIndex][0] < inputPixels[rowIndex+1, colIndex][0]/edgeThreshold) or (inputPixels[rowIndex, colIndex][0] < inputPixels[rowIndex, colIndex+1][0]/edgeThreshold):
                    outputPixels[rowIndex, colIndex] = edgeColor
                else:
                    outputPixels[rowIndex, colIndex] = backgroundColor
                    #if(inputPixels[rowIndex, colIndex][0] < 100):
                    #    outputPixels[rowIndex, colIndex] = shadeColor
                    #else:
                    #    outputPixels[rowIndex, colIndex] = backgroundColor
                    #outputPixels[rowIndex, colIndex] = backgroundColor
    print("Edge detection complete")
    return outputPixels

#Saves the image with 1 bit per channel
def saveImage(imageName):
    img = Image.fromarray(pixelArray, 'RGBA')
    img = img.convert("P", palette=Image.ADAPTIVE, colors=8)
    # Save the image
    img.save(imageName, optimize=True)
    print(f"Image saved as ",imageName)

def saveImage2(imageName):
    img = Image.fromarray(pixelArray, 'RGBA')
    # Save the image
    img.save(imageName, optimize=True)
    print(f"Image saved as ",imageName)


inputType = sys.argv[1]
loadedImage = sys.argv[2]
outputPath = sys.argv[3]
desiredProcess = sys.argv[4]

#PNG
if inputType == "image":
    loadImage(loadedImage)
    if desiredProcess == "dither":
        pixelArray = posterizeDither(2, pixelArray)
        saveImage(outputPath)
    elif desiredProcess == "posterize":
        posterize(8)
        saveImage(outputPath)
    elif desiredProcess == "edge-detect":
        pixelArray = blur(2, pixelArray)
        pixelArray = edgeDetection(1.2, pixelArray)
        saveImage2(outputPath)
    elif desiredProcess == "blur":
        pixelArray = blur(2, pixelArray)
        saveImage2(outputPath)
    else:
        print("Please enter either 'dither' or 'posterize'")
#mp4
elif inputType == "video":
    if desiredProcess == "dither":
        ditherVideo(loadedImage)
    else:
        print("Only video dithering is currently supported")
#Gif
elif inputType == "gif":
    if desiredProcess == "dither":
        ditherGif(loadedImage)
    else:
        print("Only video dithering is currently supported")

print("Peak memory usage: ",(p.memory_info().peak_wset/1000)/1000,"MB")