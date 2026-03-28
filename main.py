import image_processes as ip

import sys
import psutil
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *



class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.root.geometry("800x600")
        self.process_list = [
            ip.ResizeProcess(),
            ip.SobelEdgeDetectionProcess(),
            ip.OrderedDitherProcess(),
        ]
        self.input_path = tk.StringVar()
        self.setup_ui()


    # =========================
    # UI Setup
    # =========================
    def setup_ui(self):
        self.setup_menu()
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        # Process Pipeline (Top)
        processes_panel = ttk.Frame(main_frame, width=250, bootstyle="dark")
        processes_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)
        process_panel_label = tk.Label(processes_panel, text="Process Pipeline")
        process_panel_label.pack(fill=X, padx=2, pady=2)

        tk.Button(processes_panel, text="Save", command = self.do_it).pack(fill=X, padx=10, pady=10)
        tk.Button(processes_panel, text="Add Process", command = NONE).pack(fill=X, padx=2, pady=2)

        # Process Pipeline Stack
        self.process_stack_frame = ttk.Frame(processes_panel, bootstyle="secondary")
        self.process_stack_frame.pack(fill=BOTH, padx=4, pady=4)
        self.refresh_process_frames()


        # Image Preview
        self.preview_panel = ttk.Frame(main_frame)
        self.preview_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        self.image_label = ttk.Label(self.preview_panel, anchor="center")
        self.image_label.pack(fill=BOTH, expand=True)



    # =========================
    # Top Menu
    # =========================
    def setup_menu(self):
        menu_bar = Menu(self.root)

        file_menu = Menu(menu_bar, tearoff = 0)
        menu_bar.add_cascade(label ='File', menu = file_menu)
        
        file_menu.add_command(label ='Import...', command = self.open_image_selection)
        file_menu.add_command(label ='Export...', command = None)
        file_menu.add_separator()
        file_menu.add_command(label ='Exit', command = self.root.destroy)

        processes_menu = Menu(menu_bar, tearoff = 0)
        menu_bar.add_cascade(label ='Add Image Process', menu = processes_menu)
        processes_menu.add_command(label ='Dither', command = None)
        processes_menu.add_command(label ='Resize', command = None)

        help_menu = Menu(menu_bar, tearoff = 0)
        menu_bar.add_cascade(label ='Help', menu = help_menu)
        help_menu.add_command(label='About Image Proccesses')
        help_menu.add_command(label='What is <program_name>?')
        self.root.config(menu=menu_bar)
    

    # =========================
    # Pipeline Logic
    # =========================
    def create_process_frame(self, parent_frame, process, index):
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

        tk.Button(buttons_frame, text="✕", command=lambda i=index: self.remove_process(i)).pack(fill=X, padx=2, pady=(2,0))
        tk.Button(buttons_frame, text="▲", command=lambda p=index: self.move_up(p)).pack(fill=X, padx=2, pady=2)
        tk.Button(buttons_frame, text="▼", command=lambda p=index: self.move_down(p)).pack(fill=X, padx=2, pady=(0,2))

    def refresh_process_frames(self):
        for widget in self.process_stack_frame.winfo_children():
            widget.destroy()

        for i, process in enumerate(self.process_list):
            self.create_process_frame(self.process_stack_frame, process, i)

    def add_process(self, process):
        self.process_list.append(process)
    
    def remove_process(self, index):
        del self.process_list[index]
        self.refresh_process_frames()


    def move_up(self, index):
        if index > 0:
            self.process_list[index], self.process_list[index-1] = self.process_list[index-1], self.process_list[index]
            self.refresh_process_frames()

    def move_down(self, index):
        
        if index < len(self.process_list) - 1:
            self.process_list[index], self.process_list[index+1] = self.process_list[index+1], self.process_list[index]
            self.refresh_process_frames()



    # =========================
    # Pipeline IO
    # =========================
    def open_image_selection(self):
        path = filedialog.askopenfilename()
        if path:  # user didn't cancel
            self.input_path.set(path)

    def do_it(self):

        print(self.input_path.get())
        img_pixels = ip.load_image(self.input_path.get())
        

        for process in self.process_list:
            
            img_pixels = process.perform_process(img_pixels)

        print(img_pixels)
            
        ip.save_image("image-output/idk.png", img_pixels)
 


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style("superhero")
    app = ImageEditorApp(root)
    root.mainloop()