import os
import sys
from PIL import Image
import random
import time
import numpy as np

loadedImage = sys.argv[1]

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

def modifyImage():
    print(pixelArray[0])
 
    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):
            #print("Pixel: ",pixelArray[rowIndex, colIndex][3])
            r = pixelArray[rowIndex, colIndex][0]
            g = pixelArray[rowIndex, colIndex][1]
            b = pixelArray[rowIndex, colIndex][2]
            
            #if(g > r*1 and g > b*1):
                #pixelArray[rowIndex, colIndex] = (r*2, g/2, b, 255)

            pixelArray[rowIndex, colIndex][0] = randomInt
            pixelArray[rowIndex, colIndex][1] = g
            pixelArray[rowIndex, colIndex][2] = b
            
            #pixelArray[rowIndex, colIndex] = (128,128,128, 255)
            #pixelArray[rowIndex, colIndex][0] = 255
    #pixelArray[0:100, 0:100, 0] = 0
    
def edgeDetect():
    tempArray = np.copy(pixelArray)

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):
            pixelArray[rowIndex, colIndex] = (200,200,200,255)
            #print("Pixel: ",pixelArray[rowIndex, colIndex][3])
            #thisPixel = [rowIndex, colIndex]

            
            
            if(rowIndex >= 3 and rowIndex <= len(pixelArray) -3 and colIndex >=3 and colIndex <= len(pixelArray[0]) - 3):
                #pixelArray[rowIndex, colIndex] = (combinedRGB, combinedRGB, combinedRGB, 255)

                r = tempArray[rowIndex, colIndex][0]
                g = tempArray[rowIndex, colIndex][1]
                b = tempArray[rowIndex, colIndex][2]
                combinedRGB = (r + b + g)/3
                
                r = tempArray[rowIndex-1 , colIndex][0]
                g = tempArray[rowIndex-1, colIndex][1]
                b = tempArray[rowIndex-1, colIndex][2]
                
                neighorCombinedRGB1 = (r + b + g)/3

                r = tempArray[rowIndex-2 , colIndex][0]
                g = tempArray[rowIndex-2, colIndex][1]
                b = tempArray[rowIndex-2, colIndex][2]
                
                neighorCombinedRGB2 = (r + b + g)/3

                r = tempArray[rowIndex+1 , colIndex][0]
                g = tempArray[rowIndex+1, colIndex][1]
                b = tempArray[rowIndex+1, colIndex][2]
                
                neighorCombinedRGB3 = (r + b + g)/3

                r = tempArray[rowIndex+2 , colIndex][0]
                g = tempArray[rowIndex+2, colIndex][1]
                b = tempArray[rowIndex+2, colIndex][2]
                
                neighorCombinedRGB4 = (r + b + g)/3



                r = tempArray[rowIndex , colIndex-1][0]
                g = tempArray[rowIndex, colIndex-1][1]
                b = tempArray[rowIndex, colIndex-1][2]
                
                neighorCombinedRGB5 = (r + b + g)/3

                r = tempArray[rowIndex , colIndex-2][0]
                g = tempArray[rowIndex, colIndex-2][1]
                b = tempArray[rowIndex, colIndex-2][2]
                
                neighorCombinedRGB6 = (r + b + g)/3

                r = tempArray[rowIndex , colIndex+1][0]
                g = tempArray[rowIndex, colIndex+1][1]
                b = tempArray[rowIndex, colIndex+1][2]
                
                neighorCombinedRGB7 = (r + b + g)/3

                r = tempArray[rowIndex , colIndex+2][0]
                g = tempArray[rowIndex, colIndex+2][1]
                b = tempArray[rowIndex, colIndex+2][2]
                
                neighorCombinedRGB8 = (r + b + g)/3

                neighorCombinedRGB = (neighorCombinedRGB4+neighorCombinedRGB4+neighorCombinedRGB4+neighorCombinedRGB4+neighorCombinedRGB5+neighorCombinedRGB6+neighorCombinedRGB7+neighorCombinedRGB8)/8
                


                if(combinedRGB > neighorCombinedRGB + 30):
                    pixelArray[rowIndex, colIndex] = (0,0,0,255)
               

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


def posterize(numberOfShades):
    tempArray = np.copy(pixelArray)
    
    totalR = []
    totalG = []
    totalB = []

    r = 0
    g = 0
    b = 0

    lowestBlackWhite = 255
    highestBlackWhite = 0


    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):

            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]

            totalR.append(r)
            totalG.append(g)
            totalB.append(b)

            blackWhite = int((0.299*r)+(0.857*g)+(0.114*b))
            if(blackWhite > highestBlackWhite):
                highestBlackWhite = np.clip(blackWhite, 0, 255)
            elif(blackWhite < lowestBlackWhite):
                lowestBlackWhite = np.clip(blackWhite, 0, 255)


    meanR = int(np.mean(totalR))
    meanG = int(np.mean(totalG))
    meanB = int(np.mean(totalB))


    meanBlackWhite = int((0.299*meanR)+(0.857*meanG)+(0.114*meanB))

    #meanR = meanBlackWhite
    #meanG = meanBlackWhite
    #meanB = meanBlackWhite

    
    blackWhiteRange = highestBlackWhite - lowestBlackWhite
    shadeSize = int(blackWhiteRange/numberOfShades)
    shades = []
    shadeMultipliers = []

    #Calculate shade values
    for shade in range(numberOfShades + 1):
        shades.append(blackWhiteRange)
        blackWhiteRange -= shadeSize

    #Calculate shade multiplers
    for shade in shades:
        #scale = 1-(count/numberOfShades)
        #multiplier = highestBlackWhite * scale + lowestBlackWhite * (1-scale)
        #shadeMultipliers.append(multiplier/meanBlackWhite)
        shadeMultipliers.append(meanBlackWhite/np.clip(shade, 1, 255))
    
    print("Shade multipliers: ", shadeMultipliers)


    print("Shadesize: ",shadeSize)
    print("Highest: ",highestBlackWhite)
    print("Lowest",lowestBlackWhite)
    print("Average:",meanBlackWhite)
    print(shades)


    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):
            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]
            combinedRGB = int((0.299*r)+(0.857*g)+(0.114*b))

            #print("Start of shade shit")
            #print("Shade index")
            for shadeIndex, shade in enumerate(shades):
                joe = numberOfShades - shadeIndex + 1
                #print(joe)
                if(combinedRGB >= shade):
                    r = np.clip(int(meanR/shadeMultipliers[shadeIndex]),0,255)
                    g = np.clip(int(meanG/shadeMultipliers[shadeIndex]),0,255)
                    b = np.clip(int(meanB/shadeMultipliers[shadeIndex]),0,255)
                    pixelArray[rowIndex, colIndex] = (r, g, b, 255)
                    break

            
                #pixelArray[rowIndex, colIndex] = (int(combinedRGB*0.7), int(combinedRGB*0.9), int(combinedRGB*1.1), 255)         

            #pixelArray[rowIndex, colIndex] = (combinedRGB, combinedRGB, combinedRGB, 255)         












def posterize2(numOfColors):
    stepSize = round(256/(numOfColors-1))
    tempArray = np.copy(pixelArray)
    print(stepSize)
    r = 0
    g = 0
    b = 0

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

#Ordered dithering
def posterizeDither(numOfColors):
    stepSize = round(256/(numOfColors-1))
    tempArray = np.copy(pixelArray)
    print(stepSize)

    rule = np.array([[0.0, 0.5], [0.75, 0.25]])

    r = 0
    g = 0
    b = 0

    print("stepSize*0.25: ",stepSize*0.25)
    print("stepSize*0.5: ",stepSize*0.5)
    print("stepSize*0.75: ",stepSize*0.75)

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):



            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]
            #print("Before: ",r,", ",g,", ",b)

            #r = np.clip((round(r/ stepSize) * stepSize), 0, 255)
            #g = np.clip((round(g/ stepSize) * stepSize), 0, 255)
            #b = np.clip((round(b/ stepSize) * stepSize), 0, 255)
            #print("After: ",r,", ",g,", ",b)
            
            newR = 255
            newG = 255
            newB = 255
            
            newR = dither(r, rowIndex, colIndex, stepSize)
            newG = dither(g, rowIndex, colIndex, stepSize)
            newB = dither(b, rowIndex, colIndex, stepSize)

            pixelArray[rowIndex, colIndex] = (newR, newG, newB, 255)   

def dither(color, rowIndex, colIndex, stepSize):
    newColor = 0
    if(rowIndex % 2 == 0):
        if(colIndex % 2 == 0):
            if( color > (round(stepSize*0.25))):
                newColor = 255
        else:
            if(color > (round(stepSize*0.75))):
                newColor = 255
    else:
        if(colIndex % 2 == 0):
            if(color > (round(stepSize*0.5))):
                newColor = 255

               
        else:
            if(color > (round(stepSize*0.25))):
                newColor = 255
    return newColor
    

def saveModifiedImage():
    img = Image.fromarray(pixelArray, 'RGBA')
    # Save the image
    img.save("new-image.png")
    print(f"Image saved as new-image.png")



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

def saveImage(filename="random_noise.png"):
    # Convert the matrix to a NumPy array
    np_array = np.array(matrix, dtype=np.uint8)
    # Create an image from the array
    img = Image.fromarray(np_array, 'RGB')
    # Save the image
    img.save(filename)
    print(f"Image saved as {filename}")


loadImage(loadedImage)
#loadImage("images.jpg")
#modifyImage()
#blur(2)
#posterize(5)
#posterize2(2)
posterizeDither(2)
saveModifiedImage()
#saveImage()
