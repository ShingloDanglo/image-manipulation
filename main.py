import image_processes as ip

import sys
import psutil
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk



class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.root.geometry("800x600")
        self.process_list = []
        self.available_processes = [
            ip.ResizeProcess(),
            ip.SobelEdgeDetectionProcess(),
            ip.OrderedDitherProcess(),
            ip.PosterizeProcess(),
            ip.MakeSeamlessProcess(),
            ip.BoxBlurProcess(),
            ip.GaussianBlurProcess(),
            ip.AdjustHSVProcess()
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

        ttk.Button(processes_panel, text="Save", command = self.export_image).pack(fill=X, padx=10, pady=4)
        ttk.Button(processes_panel, text="Add Process", command = self.add_process_dialog).pack(fill=X, padx=10, pady=4)
        ttk.Button(processes_panel, text="Update Preview", command = self.update_preview).pack(fill=X, padx=10, pady=4)

        # Process Pipeline Stack
        self.process_stack_frame = ttk.Frame(processes_panel, bootstyle="secondary")
        self.process_stack_frame.pack(fill=BOTH, padx=4, pady=4)
        self.refresh_process_frames()


        # Image Preview
        self.image_original = None
        self.image_ratio = None
        self.image_tk = None

        self.preview_canvas = tk.Canvas(main_frame)
        self.preview_canvas.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0,10), pady=10)
        
        self.canvas_image = self.preview_canvas.create_image(
            0, 0,
            anchor="nw",
            image=None
        )

        self.preview_canvas.bind('<Configure>', self.fit_image)


    def fit_image(self, event):
        self.resize_preview(event.width, event.height)
        

    def resize_preview(self, width, height):
        if(self.image_original is None):
            return

        canvas_ratio = width / height

        new_width = 0
        new_height = 0

        if(canvas_ratio > self.image_ratio):
            new_height = int(height)
            new_width = int(height * self.image_ratio) 
        else:
            new_width = int(width) 
            new_height = int(width / self.image_ratio)


        resized_image = self.image_original.resize((new_width, new_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        self.preview_canvas.itemconfig(self.canvas_image, image=self.image_tk)

        # Center image
        self.preview_canvas.coords(
            self.canvas_image
        )
    


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

        help_menu = Menu(menu_bar, tearoff = 0)
        menu_bar.add_cascade(label ='Help', menu = help_menu)
        help_menu.add_command(label='About Image Proccesses')
        help_menu.add_command(label='What is <program_name>?')
        self.root.config(menu=menu_bar)

    # =========================
    # Add Process Dialogue
    # =========================
    def add_process_dialog(self):
        dialog = ttk.Toplevel(self.root)
        dialog.title("Select Process")
        dialog.transient(self.root)
        dialog.grab_set()

        processs_listbox = tk.Listbox(dialog)
        for process in self.available_processes:
            processs_listbox.insert(tk.END, process.process_name)
        processs_listbox.pack(fill=tk.BOTH, expand=True, padx=10)

        def on_add():
            process_index = 0
            try:
                process_index = processs_listbox.curselection()[0]
            except IndexError:
                print("Error selecting process to add:\n"+IndexError)
                dialog.destroy()
            
            # Create instance of same class as selected prcoess
            process_class = type(self.available_processes[process_index])
            new_process = process_class()
            self.add_process(new_process)            
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        # OK / Cancel buttons
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=10)
        
        cancel_button = ttk.Button(
            buttons_frame,
            text="Cancel",
            command=on_cancel,
            bootstyle="danger"
        )
        cancel_button.configure(padding=(0,0))
        cancel_button.pack(side=LEFT, padx=5)

        add_button = ttk.Button(
            buttons_frame,
            text="Add",
            command=on_add,
            bootstyle="success"
        )
        add_button.configure(padding=(0,0))
        add_button.pack(side=RIGHT, padx=5)

        

    # =========================
    # Pipeline Logic
    # =========================
    def create_process_frame(self, parent_frame, process, index):
        process_frame = ttk.Frame(parent_frame)
        process_frame.pack(fill=BOTH, padx=10, pady=5)

        # Generate process fields
        fields_frame = ttk.Frame(process_frame, bootstyle="dark")
        fields_frame.pack(side=LEFT, fill=BOTH, expand=True)

        tk.Label(fields_frame, text=process.process_name).pack(fill=X, padx=4, pady=4)

        for user_input in process.user_inputs:
            row = ttk.Frame(fields_frame)
            row.pack(fill=X, padx=4, pady=(0,4))

            tk.Label(row, text=user_input.label).pack(side=LEFT, fill=X, expand=True, pady=(0, 4))
            
            input_widget = self.build_input_widget(row, user_input)
            input_widget.pack(side=RIGHT, fill=X, pady=(0, 4))

        # Buttons
        buttons_frame = ttk.Frame(process_frame)
        buttons_frame.pack(side=RIGHT, anchor="n", padx=2, pady=2)

        delete_button = ttk.Button(buttons_frame, text="✕",command=lambda i=index: self.remove_process(i))
        delete_button.configure(bootstyle='danger', padding=(0,0))
        delete_button.pack(fill=X, padx=2, pady=(2,0))

        move_up_button = ttk.Button(buttons_frame, text="▲", command=lambda p=index: self.move_up(p))
        move_up_button.configure(padding=(0,0))
        move_up_button.pack(fill=X, padx=2, pady=(2,0))

        move_down_button = ttk.Button(buttons_frame, text="▼", command=lambda p=index: self.move_down(p))
        move_down_button.configure(padding=(0,0))
        move_down_button.pack(fill=X, padx=2, pady=(2,0))

    def build_input_widget(self, parent_frame, user_input):
        if(isinstance(user_input, ip.IntegerInput)):
            return ttk.Entry(parent_frame, textvariable=user_input.value)
        elif(isinstance(user_input, ip.DoubleInput)):
            return ttk.Entry(parent_frame, textvariable=user_input.value)
        elif(isinstance(user_input, ip.IntegerSliderInput)):
            return tk.Scale(
                parent_frame,
                from_=user_input.min_value,
                to=user_input.max_value,
                orient=tk.HORIZONTAL,
                variable=user_input.value,
                resolution=1
            )
        elif(isinstance(user_input, ip.DoubleSliderInput)):
            return tk.Scale(
                parent_frame,
                from_=user_input.min_value,
                to=user_input.max_value,
                orient=tk.HORIZONTAL,
                variable=user_input.value,
                resolution=0.01
            )
        else:
            print("user_input did not match any input classes")
            return ttk.Entry(parent_frame, textvariable=user_input.value.get())
    
    
    def refresh_process_frames(self):
        for widget in self.process_stack_frame.winfo_children():
            widget.destroy()

        for i, process in enumerate(self.process_list):
            self.create_process_frame(self.process_stack_frame, process, i)

    def add_process(self, process):
        self.process_list.append(process)
        self.refresh_process_frames()
    
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
        if not path:
            return
        
        self.input_path.set(path)

        # Update preview
        self.image_original = Image.open(path)
        self.image_ratio = (
            self.image_original.size[0] /
            self.image_original.size[1]
        )

        self.resize_preview(
            self.preview_canvas.winfo_width(),
            self.preview_canvas.winfo_height()
        )

    def update_preview(self):
        img_pixels = ip.load_image(self.input_path.get())

        for process in self.process_list:
            img_pixels = process.perform_process(img_pixels)

        self.image_original = Image.fromarray(img_pixels)
        self.image_ratio = (
            self.image_original.size[0] /
            self.image_original.size[1]
        )

        self.resize_preview(
            self.preview_canvas.winfo_width(),
            self.preview_canvas.winfo_height()
        )

    
    def export_image(self):
        export_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="output.png"
        )
        if not export_path:
            return

        print(export_path)
        img_pixels = ip.load_image(self.input_path.get())

        for process in self.process_list:
            img_pixels = process.perform_process(img_pixels)
            
        ip.save_image(export_path, img_pixels)

    
        
    


if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = ImageEditorApp(root)
    root.mainloop()