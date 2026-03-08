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

@njit
def generate_gaussian_kernel(size, sigma):
    """ Outputs a 1d gaussian kernel """
    radius = size // 2
    
    kernel = np.zeros(size)

    for i in range(size):
        x = i - radius
        kernel[i] = np.exp(-(x*x) / (2*sigma*sigma))

    kernel /= np.sum(kernel)
    return kernel

# =========================
# Image effects
# =========================

@njit(parallel=True)
def dither_pixel(color,step_size, dither_threshold):
    quantized_color = round(color//step_size * step_size)
    
    if(color % step_size) > dither_threshold:
        quantized_color = min(quantized_color + step_size, 255)
    return quantized_color


#Ordered dithering
@njit(parallel=True)
def ordered_dither(input_pixels, color_steps=2):
    width, height, _ = input_pixels.shape
    output_pixels = np.empty_like(input_pixels)

    step_size = int(256/(color_steps-1))

    matrix_size = 8
    #2x2 matrix
    if(matrix_size == 2):
        rule = step_size * (1.0 / 4.0) * (np.array([
            [1, 3],
            [4, 2]
        ]) - 0.5)
    #4x4 matrix
    elif(matrix_size == 4):
        rule = step_size * (1.0 / 16.0) * (np.array([
            [1, 9, 3, 11],
            [13, 5, 15, 7],
            [4, 12, 2, 10],
            [16, 8, 14, 6]
        ]) - 0.5)
    #8x8 matrix
    else:
        rule = step_size * (1.0 / 64.0) * (np.array([
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

            dither_threshold = rule[x % matrix_size, y % matrix_size]
            r, g, b, a = input_pixels[x, y]

            output_pixels[x, y, 0] = dither_pixel(r, step_size, dither_threshold)
            output_pixels[x, y, 1] = dither_pixel(g, step_size, dither_threshold)
            output_pixels[x, y, 2] = dither_pixel(b, step_size, dither_threshold)
            output_pixels[x, y, 3] = dither_pixel(a, step_size, dither_threshold)
            
    print("Dither applied")
    return output_pixels

@njit(parallel=True)
def posterize(input_pixels, color_steps):
    stepSize = int(256/(color_steps-1))
    width, height, _ = input_pixels.shape
    output_pixels = np.empty_like(input_pixels)

    for y in prange(height):
        for x in range(width):
            r, g, b, a = input_pixels[x, y]

            output_pixels[x, y, 0] = clip((round(r/ stepSize) * stepSize), 0, 255)
            output_pixels[x, y, 1] = clip((round(g/ stepSize) * stepSize), 0, 255)
            output_pixels[x, y, 2] = clip((round(b/ stepSize) * stepSize), 0, 255)
            output_pixels[x, y, 3] = clip((round(a/ stepSize) * stepSize), 0, 255)

    return output_pixels


@njit(parallel=True)
def sobel_edge_detect(input_pixels):
    width, height, _ = input_pixels.shape
    output_pixels = np.empty_like(input_pixels)
    
    x_matrix = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    y_matrix = np.array([
        [-1, -2,-1],
        [0, 0,0],
        [1, 2, 1]
    ])

    for y in prange(height):
        for x in range(width):

            x_total = 0
            y_total = 0

            for ky in range(3):
                for kx in range(3):
                    nx = clip(x + kx - 1, 0, width - 1)
                    ny = clip(y + ky - 1, 0, height - 1)

                    pixel = input_pixels[nx, ny, 0]

                    x_total += pixel * x_matrix[kx, ky]
                    y_total += pixel * y_matrix[kx, ky]

            magnitude = round((x_total * x_total + y_total * y_total) ** 0.5)

            if(magnitude > 128):
                output_pixels[x, y] = (0, 0, 0, 255)
            else:
                output_pixels[x, y] = (255, 255, 255, 255)
                    
    print("Edge detection complete")
    return output_pixels



@njit(parallel=True)
def box_blur(input_pixels, kernel_size):
    width, height, _ = input_pixels.shape
    output_pixels = np.copy(input_pixels)

    for y in prange(height):
        for x in range(width):
            r_sum = 0
            g_sum = 0
            b_sum = 0
            a_sum = 0
            count = 0

            x0 = clip(x - kernel_size, 0, width - 1)
            x1 = clip(x + kernel_size, 0, width - 1)

            y0 = clip(y - kernel_size, 0, height - 1)
            y1 = clip(y + kernel_size, 0, height - 1)

            # Loop through neighboring pixels
            for ny in range(y0, y1 + 1):
                for nx in range(x0, x1 + 1):
                    pixel = input_pixels[nx, ny]
                    r_sum += pixel[0]
                    g_sum += pixel[1]
                    b_sum += pixel[2]
                    a_sum += pixel[3]
                    count += 1


            output_pixels[x, y, 0] = r_sum // count
            output_pixels[x, y, 1] = g_sum // count
            output_pixels[x, y, 2] = b_sum // count
            output_pixels[x, y, 3] = a_sum // count

    print("Image box blurred")
    return output_pixels


@njit(parallel=True)
def gaussian_blur(input_pixels, kernel_size, sigma):
    width, height, _ = input_pixels.shape
    radius = kernel_size // 2
    
    # Generate 1d gaussian kernel
    kernel = np.zeros(kernel_size)

    for i in range(kernel_size):
        x = i - radius
        kernel[i] = np.exp(-(x*x) / (2*sigma*sigma))

    kernel /= np.sum(kernel)


    temp = np.copy(input_pixels)
    output_pixels = np.copy(input_pixels)

    # Horizontal pass
    for y in prange(height):
        for x in range(width):

            r = 0.0
            g = 0.0
            b = 0.0

            for k in range(-radius, radius + 1):
                nx = clip(x + k, 0, width - 1)
                weight = kernel[k + radius]

                pixel = input_pixels[nx, y]

                r += pixel[0] * weight
                g += pixel[1] * weight
                b += pixel[2] * weight

            temp[x, y, 0] = int(r)
            temp[x, y, 1] = int(g)
            temp[x, y, 2] = int(b)
            temp[x, y, 3] = 255


    # Vertical pass
    for y in prange(height):
        for x in range(width):

            r = 0.0
            g = 0.0
            b = 0.0

            for k in range(-radius, radius + 1):
                ny = clip(y + k, 0, height - 1)
                weight = kernel[k + radius]

                pixel = temp[x, ny]

                r += pixel[0] * weight
                g += pixel[1] * weight
                b += pixel[2] * weight

            output_pixels[x, y, 0] = int(r)
            output_pixels[x, y, 1] = int(g)
            output_pixels[x, y, 2] = int(b)
            output_pixels[x, y, 3] = 255

    print("Image gaussian blurred")

    return output_pixels




@njit(parallel=True)
def make_seamless(input_pixels, num):
    width, height, _ = input_pixels.shape
    output_pixels = np.copy(input_pixels)

    num = round(width / 4)

    mid_point = round(width/2)
    for y in prange(height):
        for x in range(num):
            mix_multiplier = ((num -x) + 1) / num

            # Move the left half of the middle section to the right side of the image
            output_pixels[width - 1 - x, y, 0] = round(input_pixels[mid_point - x, y, 0] * mix_multiplier) +  round(input_pixels[width - 1 - x, y, 0] * (1 - mix_multiplier ))
            output_pixels[width  - 1 - x, y, 1] = round(input_pixels[mid_point - x, y, 1] * mix_multiplier) +  round(input_pixels[width - 1 - x, y, 1] * (1 - mix_multiplier ))
            output_pixels[width  - 1 - x, y, 2] = round(input_pixels[mid_point - x, y, 2] * mix_multiplier) +  round(input_pixels[width - 1 - x, y, 2] * (1 - mix_multiplier ))

            # Move the right half of the middle section to the left side of the image
            output_pixels[x, y, 0] = round(input_pixels[mid_point + x, y, 0] * mix_multiplier) +  round(input_pixels[x, y, 0] * (1 - mix_multiplier ))
            output_pixels[x, y, 1] = round(input_pixels[mid_point + x, y, 1] * mix_multiplier) +  round(input_pixels[x, y, 1] * (1 - mix_multiplier ))
            output_pixels[x, y, 2] = round(input_pixels[mid_point + x, y, 2] * mix_multiplier) +  round(input_pixels[x, y, 2] * (1 - mix_multiplier ))

    input_pixels = np.copy(output_pixels)

    num = round(height / 4)

    mid_point = round(height/2)
    for x in prange(width):
        for y in range(num):
            mix_multiplier = ((num -y) + 1) / num

            # Move the top half of the middle section to the bottom of the image
            output_pixels[x, height - 1 - y, 0] = round(input_pixels[x, mid_point - y, 0] * mix_multiplier) +  round(input_pixels[x, height - 1 - y, 0] * (1 - mix_multiplier ))
            output_pixels[x, height - 1 - y, 1] = round(input_pixels[x, mid_point - y, 1] * mix_multiplier) +  round(input_pixels[x, height - 1 - y, 1] * (1 - mix_multiplier ))
            output_pixels[x, height - 1 - y, 2] = round(input_pixels[x, mid_point - y, 2] * mix_multiplier) +  round(input_pixels[x, height - 1 - y, 2] * (1 - mix_multiplier ))

            # Move the bottom half of the middle section to the top of the image
            output_pixels[x, y, 0] = round(input_pixels[x, mid_point + y, 0] * mix_multiplier) +  round(input_pixels[x, y, 0] * (1 - mix_multiplier ))
            output_pixels[x, y, 1] = round(input_pixels[x, mid_point + y, 1] * mix_multiplier) +  round(input_pixels[x, y, 1] * (1 - mix_multiplier ))
            output_pixels[x, y, 2] = round(input_pixels[x, mid_point + y, 2] * mix_multiplier) +  round(input_pixels[x, y, 2] * (1 - mix_multiplier ))
 

    return output_pixels


# =========================
# Video
# =========================

# TODO: Rewrite video functions to reduce memory usage. Currently, even short HD clips
# use several GB of memory

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
        "dither": lambda img_pixels: ordered_dither(img_pixels, 4),
        "posterize": lambda img_pixels: posterize(img_pixels, 4),
        "sobel-edge-detect": lambda img_pixels: sobel_edge_detect(img_pixels),
        "box-blur": lambda img_pixels: box_blur(img_pixels, 1),
        "gaussian-blur": lambda img_pixels: gaussian_blur(img_pixels, 2, 2),
        "resize": lambda img_pixels: resize_image(img_pixels, 1024, 1024),
        "make-seamless": lambda img_pixels: make_seamless(img_pixels, 100),
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
        process_video(input_path, output_path, processors[actions[0]])

    elif input_type == "gif":
        process_gif(input_path, output_path, processors[action])   


if __name__ == "__main__":
    main()