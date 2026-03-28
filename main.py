import image_processes as ip

import sys
import psutil
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *









# =========================
# Main CLI
# =========================



def main():
    process_list = []
    process_list.append(ip.resize_process())
    process_list.append(ip.sobel_edge_detection_process())
    process_list.append(ip.ordered_dither_process())


    def create_process_frame(parent_frame, process, index):
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

        tk.Button(buttons_frame, text="✕", command=lambda i=index: remove_process(i)).pack(fill=X, padx=2, pady=(2,0))
        tk.Button(buttons_frame, text="▲", command=lambda p=process_list[index]: move_up(p)).pack(fill=X, padx=2, pady=2)
        tk.Button(buttons_frame, text="▼", command=lambda p=process_list[index]: move_down(p)).pack(fill=X, padx=2, pady=(0,2))

    def refresh_processes():
        for widget in proccess_stack_frame.winfo_children():
            widget.destroy()

        for i, process in enumerate(process_list):
            create_process_frame(proccess_stack_frame, process, i)

    def add_process(process):
        process_list.append(process)
    
    def remove_process(index):
        del process_list[index]
        refresh_processes()


    def move_up(process):
        index = process_list.index(process)
        if index > 0:
            process_list[index], process_list[index-1] = process_list[index-1], process_list[index]
            refresh_processes()

    def move_down(process):
        index = process_list.index(process)
        if index < len(process_list) - 1:
            process_list[index], process_list[index+1] = process_list[index+1], process_list[index]
            refresh_processes()




    def open_image_selection():
        path = filedialog.askopenfilename()
        if path:  # user didn't cancel
            input_path.set(path)

    def do_it():

        print(input_path.get())
        img_pixels = ip.load_image(input_path.get())
        

        for process in process_list:
            
            img_pixels = process.perform_process(img_pixels)

        print(img_pixels)
            
        ip.save_image("image-output/idk.png", img_pixels)
        



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
    input_path = tk.StringVar()
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

    tk.Button(processes_panel, text="Save", command = do_it).pack(fill=X, padx=10, pady=10)

    tk.Button(processes_panel, text="Add Process", command = NONE).pack(fill=X, padx=2, pady=2)


    # Process Pipeline Stack
    proccess_stack_frame = ttk.Frame(processes_panel, bootstyle="secondary")
    proccess_stack_frame.pack(fill=BOTH, padx=4, pady=4)

    for i, process in enumerate(process_list):
        create_process_frame(proccess_stack_frame, process, i)



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
 


if __name__ == "__main__":
    main()