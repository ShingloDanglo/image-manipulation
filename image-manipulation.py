import os
from PIL import Image
import random
import time
import numpy as np

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

                # Loop through the 5x5 neighborhood around the current pixel
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


def posterize():
    tempArray = np.copy(pixelArray)
    totalR = []
    totalG = []
    totalB = []

    r = 0
    g = 0
    b = 0

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):

            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]

            totalR.append(r)
            totalG.append(g)
            totalB.append(b)

    meanR = int(np.mean(totalR))
    meanG = int(np.mean(totalG))
    meanB = int(np.mean(totalB))

    meanBlackWhite = int((0.299*meanR)+(0.857*meanG)+(0.114*meanB))

    for rowIndex, pixelColumn in enumerate(pixelArray):
        for colIndex, pixel in enumerate(pixelColumn):
            r = tempArray[rowIndex, colIndex][0]
            g = tempArray[rowIndex, colIndex][1]
            b = tempArray[rowIndex, colIndex][2]
            
            combinedRGB = int((0.299*r)+(0.857*g)+(0.114*b))
            if(combinedRGB >= meanBlackWhite * 2):
                pixelArray[rowIndex, colIndex] = (np.clip(int(meanR*5), 0, 255) ,np.clip(int(meanG*5), 0, 255),np.clip(int(meanB*5), 0, 255),255)
            elif(combinedRGB >= meanBlackWhite * 1.5):
                pixelArray[rowIndex, colIndex] = (np.clip(int(meanR*3), 0, 255) ,np.clip(int(meanG*3), 0, 255),np.clip(int(meanB*3), 0, 255),255)
            elif(combinedRGB >= meanBlackWhite):
                pixelArray[rowIndex, colIndex] = (np.clip(int(meanR*1.5), 0, 255) ,np.clip(int(meanG*1.5), 0, 255),np.clip(int(meanB*1.5), 0, 255),255)
            elif(combinedRGB >= meanBlackWhite * 0.7):
                pixelArray[rowIndex, colIndex] = (np.clip(int(meanR*0.8), 0, 255) ,np.clip(int(meanG*0.8), 0, 255),np.clip(int(meanB*0.8), 0, 255),255)
            else:
                pixelArray[rowIndex, colIndex] = (np.clip(int(meanR*0.5), 0, 255) ,np.clip(int(meanG*0.5), 0, 255),np.clip(int(meanB*0.5), 0, 255),255)
                #pixelArray[rowIndex, colIndex] = (int(combinedRGB*0.7), int(combinedRGB*0.9), int(combinedRGB*1.1), 255)         

            #pixelArray[rowIndex, colIndex] = (combinedRGB, combinedRGB, combinedRGB, 255)         

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


loadImage("motorbike.png")
#loadImage("images.jpg")
#modifyImage()
#blur(2)
posterize()
saveModifiedImage()
#saveImage()
