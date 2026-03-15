import sys
import psutil
import tkinter as tk
import image_processes as ip

# =========================
# Main CLI
# =========================

def main():

    p = psutil.Process()
    input_type, input_path, output_path, *actions = sys.argv[1:]

    processors = {
        "dither": lambda img_pixels: ip.ordered_dither(img_pixels, 4),
        "posterize": lambda img_pixels: ip.posterize(img_pixels, 4),
        "sobel-edge-detect": lambda img_pixels: ip.sobel_edge_detect(img_pixels),
        "box-blur": lambda img_pixels: ip.box_blur(img_pixels, 1),
        "gaussian-blur": lambda img_pixels: ip.gaussian_blur(img_pixels, 2, 2),
        "resize": lambda img_pixels: ip.resize_image(img_pixels, 1024, 1024),
        "make-seamless": lambda img_pixels: ip.make_seamless(img_pixels, 100),
    }


    if input_type == "image":
        img_pixels = ip.load_image(input_path)

        for action in actions:
            if action not in processors:
                print("Invalid process")
                return
            img_pixels = processors[action](img_pixels)
            
        ip.save_image(output_path, img_pixels)

    elif input_type == "video":
        ip.process_video(input_path, output_path, processors[actions[0]])

    elif input_type == "gif":
        ip.process_gif(input_path, output_path, processors[action])   


if __name__ == "__main__":
    main()