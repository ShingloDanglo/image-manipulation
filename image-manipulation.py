import sys
import psutil
from PIL import Image
import numpy as np
from numba import njit, prange
from moviepy import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# =========================
# IO
# =========================

def load_image(imagePath):
    image = Image.open(imagePath)
    return np.array(image)

#Saves the image with 1 bit per channel
#def saveImage(imageName):
#    img = Image.fromarray(pixelArray, 'RGBA')
#    img = img.convert("P", palette=Image.ADAPTIVE, colors=8)

#    img.save(imageName, optimize=True)
#    print(f"Image saved as ",imageName)

def save_image(output_path, img_pixels):
    img = Image.fromarray(img_pixels, 'RGBA')
    # Save the image
    img.save(output_path, optimize=True)
    print(f"Image saved as ",output_path)

# TODO: Rewrite edge detection and blur functions

# =========================
# Utils
# =========================

@njit
def clip(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

def resize_image(input_pixels, new_width, new_height):
    img = Image.fromarray(input_pixels, 'RGBA')
    new_size = (new_width, new_height)
    img = img.resize(new_size)

    return np.array(img)


# =========================
# Image effects
# =========================

@njit
def dither_pixel(color,stepSize, ditherThreshold):
    quantizedColor = round(color//stepSize * stepSize)
    
    if(color % stepSize) > ditherThreshold:
        quantizedColor = min(quantizedColor + stepSize, 255)
    return quantizedColor


#Ordered dithering
def ordered_dither(input_pixels, colorSteps=2):
    height, width, _ = input_pixels.shape
    outputPixels = np.empty_like(input_pixels)

    stepSize = int(256/(colorSteps-1))

    matrixSize = 2
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


    for y in prange(height):
        for x in range(width):

            ditherThreshold = rule[y % matrixSize, x % matrixSize]
            r, g, b, a = input_pixels[y, x]

            outputPixels[y, x, 0] = dither_pixel(r, stepSize, ditherThreshold)
            outputPixels[y, x, 1] = dither_pixel(g, stepSize, ditherThreshold)
            outputPixels[y, x, 2] = dither_pixel(b, stepSize, ditherThreshold)
            outputPixels[y, x, 3] = dither_pixel(a, stepSize, ditherThreshold)
            
    print("Dither applied")
    return outputPixels


def posterize(input_pixels, colorSteps):
    stepSize = int(256/(colorSteps-1))
    height, width, _ = input_pixels.shape
    output_pixels = np.empty_like(input_pixels)
    

    print(stepSize)

    for y in prange(height):
        for x in range(width):
            r, g, b, a = input_pixels[y, x]

            output_pixels[y, x, 0] = clip((round(r/ stepSize) * stepSize), 0, 255)
            output_pixels[y, x, 1] = clip((round(g/ stepSize) * stepSize), 0, 255)
            output_pixels[y, x, 2] = clip((round(b/ stepSize) * stepSize), 0, 255)
            output_pixels[y, x, 3] = 255

    return output_pixels

@njit
def edge_detect(inputPixels, edgeThreshold):
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
                    if(inputPixels[rowIndex, colIndex][0] < 100):
                        outputPixels[rowIndex, colIndex] = shadeColor
                    else:
                        outputPixels[rowIndex, colIndex] = backgroundColor
                    
    print("Edge detection complete")
    return outputPixels



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


@njit
def make_seamless(input_pixels, num):
    height, width, _ = input_pixels.shape
    output_pixels = np.copy(input_pixels)

    mid_point = round(width/2)
    for y in prange(height):
        for x in range(num):
            mix_multiplier = ((num -x) + 1) / num

            # Move the left half of the middle section to the right side of the image
            output_pixels[y, width - 1 - x, 0] = round(input_pixels[y, mid_point - x, 0] * mix_multiplier) +  round(input_pixels[y,  width - 1 - x, 0] * (1 - mix_multiplier ))
            output_pixels[y, width  - 1 - x, 1] = round(input_pixels[y, mid_point - x, 1] * mix_multiplier) +  round(input_pixels[y,  width - 1 - x, 1] * (1 - mix_multiplier ))
            output_pixels[y, width  - 1 - x, 2] = round(input_pixels[y, mid_point - x, 2] * mix_multiplier) +  round(input_pixels[y,  width - 1 - x, 2] * (1 - mix_multiplier ))

            # Move the right half of the middle section to the left side of the image
            output_pixels[y, x, 0] = round(input_pixels[y, mid_point + x, 0] * mix_multiplier) +  round(input_pixels[y,  x, 0] * (1 - mix_multiplier ))
            output_pixels[y, x, 1] = round(input_pixels[y, mid_point + x, 1] * mix_multiplier) +  round(input_pixels[y,  x, 1] * (1 - mix_multiplier ))
            output_pixels[y, x, 2] = round(input_pixels[y, mid_point + x, 2] * mix_multiplier) +  round(input_pixels[y,  x, 2] * (1 - mix_multiplier ))

    input_pixels = np.copy(output_pixels)

    mid_point = round(height/2)
    for x in prange(width):
        for y in range(num):
            mix_multiplier = ((num -y) + 1) / num

            # Move the top half of the middle section to the bottom of the image
            output_pixels[height - 1 - y, x, 0] = round(input_pixels[mid_point - y, x, 0] * mix_multiplier) +  round(input_pixels[height - 1 - y,  x, 0] * (1 - mix_multiplier ))
            output_pixels[height - 1 - y, x, 1] = round(input_pixels[mid_point - y, x, 1] * mix_multiplier) +  round(input_pixels[height - 1 - y,  x, 1] * (1 - mix_multiplier ))
            output_pixels[height - 1 - y, x, 2] = round(input_pixels[mid_point - y, x, 2] * mix_multiplier) +  round(input_pixels[height - 1 - y,  x, 2] * (1 - mix_multiplier ))

            # Move the bottom half of the middle section to the top of the image
            output_pixels[y, x, 0] = round(input_pixels[mid_point + y, x, 0] * mix_multiplier) +  round(input_pixels[y,  x, 0] * (1 - mix_multiplier ))
            output_pixels[y, x, 1] = round(input_pixels[mid_point + y, x, 1] * mix_multiplier) +  round(input_pixels[y,  x, 1] * (1 - mix_multiplier ))
            output_pixels[y, x, 2] = round(input_pixels[mid_point + y, x, 2] * mix_multiplier) +  round(input_pixels[y,  x, 2] * (1 - mix_multiplier ))
 

    return output_pixels


# =========================
# Video
# =========================


def process_video(input_path, output_path, processor):
    clip = VideoFileClip(input_path)
    frames = []
    pixel_array = []

    for frame in clip.iter_frames(dtype='uint8'):
        pixel_array = np.dstack([frame, np.full(frame.shape[:2], 255, dtype="uint8")])  # Convert to RGBA
        frames.append(processor(pixel_array))

    clip = ImageSequenceClip(frames, fps=clip.fps)
    clip.write_videofile(output_path, codec="libx264")
    print(f"Video saved to {output_path}")
    
def process_gif(input_path, output_path, processor):
    clip = VideoFileClip(input_path)
    frames = []
    pixel_array = []
    
    for frame in clip.iter_frames(dtype='uint8'):
        pixel_array = np.dstack([frame, np.full(frame.shape[:2], 255, dtype="uint8")])  # Convert to RGBA
        frames.append(processor(pixel_array))

    clip = ImageSequenceClip(frames, fps=clip.fps)
    clip.write_gif(output_path)
    print(f"Video GIF to {output_path}")


# =========================
# Main CLI
# =========================

def main():

    p = psutil.Process()
    input_type, input_path, output_path, *actions = sys.argv[1:]

    processors = {
        "dither": lambda img_pixels: ordered_dither(img_pixels, 5),
        "posterize": lambda img_pixels: posterize(img_pixels, 5),
        "edge-detect": lambda img_pixels: edge_detect(1.1, blur(img_pixels, 2)),
        "blur": lambda img_pixels: blur(2, img_pixels),
        "resize": lambda img_pixels: resize_image(img_pixels, 256, 256),
        "make-seamless": lambda img_pixels: make_seamless(img_pixels, 150),
    }



    if input_type == "image":
        img_pixels = load_image(input_path)

        for action in actions:
            if action not in processors:
                print("Invalid process")
                return
            img_pixels = processors[action](img_pixels)
            
        save_image(output_path, img_pixels)

    elif input_type == "video":
        process_video(input_path, output_path, processors[action])

    elif input_type == "gif":
        process_gif(input_path, output_path, processors[action])   


if __name__ == "__main__":
    main()