import image_processes as ip

import sys
import psutil
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


def open_image_selection():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )

    return path
    

class ordered_dither_process():
    def __init__(self):    
        self.input_pixels = []
        self.process_name = 'Ordered Dither'

        self.user_inputs = {
            'Color Steps': 2
        }

    def perform_process(self):
        ip.ordered_dither(self.input_pixels, self.user_inputs.get('Color Steps'))


class resize_process():
    def __init__(self):    
        self.input_pixels = []
        self.process_name = 'Resize Image'

        self.user_inputs = {
            'Width': 512,
            'Height': 512
        }

    def perform_process(self):
        ip.ordered_dither(self.input_pixels, self.user_inputs.get('Width'), self.user_inputs.get('Height'))

class sobel_edge_detection_process():
    def __init__(self):    
        self.input_pixels = []
        self.process_name = 'Sobel Edge Detection'

        self.user_inputs = {
            'Edge Threshold (0-1)': 0.5
        }

    def perform_process(self):
        ip.sobel_edge_detect(self.input_pixels, self.user_inputs.get('Edge Threshold'))


def create_process_frame(parent_frame, process):
    process_frame = ttk.Frame(parent_frame)
    process_frame.pack(fill=BOTH, padx=10, pady=5)

    # Generate process fields
    fields_frame = ttk.Frame(process_frame, bootstyle="danger")
    fields_frame.pack(side=LEFT, fill=BOTH, expand=True)

    tk.Label(fields_frame, text=process.process_name).pack(fill=X, padx=4, pady=4)

    for key, value in process.user_inputs.items():
        row = ttk.Frame(fields_frame)
        row.pack(fill=X, padx=4)

        tk.Label(row, text=key).pack(side=LEFT, fill=X, expand=True, pady=(0, 4))
        field = tk.Entry(row)
        field.insert(0, value)
        field.pack(side=RIGHT, fill=X, pady=(0, 4))

    # Buttons
    buttons_frame = ttk.Frame(process_frame, bootstyle="warning")
    buttons_frame.pack(side=RIGHT)

    tk.Button(buttons_frame, text="✕", command = NONE).pack(fill=X, padx=2, pady=(2,0))
    tk.Button(buttons_frame, text="▲", command = NONE).pack(fill=X, padx=2, pady=2)
    tk.Button(buttons_frame, text="▼", command = NONE).pack(fill=X, padx=2, pady=(0,2))

# =========================
# Main CLI
# =========================

def main():

    process_list = []
    process_list.append(resize_process())
    process_list.append(ordered_dither_process())
    process_list.append(sobel_edge_detection_process())
    process_list.append(resize_process())
    process_list.append(ordered_dither_process())
    process_list.append(ordered_dither_process())
    process_list.append(resize_process())
    process_list.append(ordered_dither_process())
    process_list.append(ordered_dither_process())
    process_list.append(resize_process())
    process_list.append(ordered_dither_process())
    process_list.append(ordered_dither_process())


    for process in process_list:
        print(vars(process))





    root = tk.Tk()
    style = ttk.Style("superhero")

    root.geometry("800x600")
    root.title("Image Editor")


    # =========================
    # Top Menu
    # =========================
    menu_bar = Menu(root)

    file_menu = Menu(menu_bar, tearoff = 0)
    menu_bar.add_cascade(label ='File', menu = file_menu)
    file_menu.add_command(label ='Import...', command = open_image_selection)
    file_menu.add_command(label ='Export...', command = None)
    file_menu.add_separator()
    file_menu.add_command(label ='Exit', command = root.destroy)

    processes_menu = Menu(menu_bar, tearoff = 0)
    menu_bar.add_cascade(label ='Add Image Process', menu = processes_menu)
    processes_menu.add_command(label ='Dither', command = None)
    processes_menu.add_command(label ='Resize', command = None)

    help_menu = Menu(menu_bar, tearoff = 0)
    menu_bar.add_cascade(label ='Help', menu = help_menu)
    help_menu.add_command(label='About Image Proccesses')
    help_menu.add_command(label='What is <program_name>?')


    # =========================
    # Proccess Pipeline Column
    # =========================
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=BOTH, expand=True)

    # Process Panel (Left)
    processes_panel = ttk.Frame(main_frame, width=250, bootstyle="dark")
    processes_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

    process_panel_label = tk.Label(processes_panel, text="Process Pipeline")
    process_panel_label.pack(fill=X, padx=2, pady=2)

    tk.Button(processes_panel, text="Add Process", command = NONE).pack(fill=X, padx=2, pady=2)


    # Process Pipeline Stack
    proccess_stack_frame = ttk.Frame(processes_panel, bootstyle="secondary")
    proccess_stack_frame.pack(fill=BOTH, padx=4, pady=4)

    for process in process_list:
        create_process_frame(proccess_stack_frame, process)


    





    # =========================
    # Image Preview Column
    # =========================

    preview_panel = ttk.Frame(main_frame)
    preview_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

    image_label = ttk.Label(preview_panel, anchor="center")
    image_label.pack(fill=BOTH, expand=True)


    
    root.config(menu = menu_bar)
    
    
    root.mainloop()

   
   
   
   
   
   
   
   
   
   
   
   
   
    p = psutil.Process()
    input_type, input_path, output_path, *actions = sys.argv[1:]

    processors = {
        "dither": lambda img_pixels: ip.ordered_dither(img_pixels, 4),
        "posterize": lambda img_pixels: ip.posterize(img_pixels, 4),
        "sobel-edge-detect": lambda img_pixels: ip.sobel_edge_detect(img_pixels, 0.5),
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