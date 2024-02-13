import tkinter as tk
from tkinter import ttk
import json
import argparse
import platform

from module.file_op import check_file

class SliderWindow:
    def __init__(self, root, data_template):
        self.root = root
        self.root.title("Weight Editor")
        self.data_template = data_template

        self.setup_main_frame()
        self.setup_scrollable_area()
        self.load_data()
        self.create_sliders_and_entries()
        self.setup_buttons()

    def setup_main_frame(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    
    def setup_scrollable_area(self):
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox(self.scrollable_window)))

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind the canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        # Update the width of the scrollable_frame to match the canvas width, accounting for scrollbar width
        canvas_width = event.width
        self.canvas.itemconfig(self.scrollable_window, width=canvas_width - 20)  # Adjust the 20 if the scrollbar width is different


    def on_frame_configure(self, event=None):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        # You may need to adjust this depending on the OS as described earlier
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    
    def load_data(self):
        with open(f'./{self.data_template}/weight.json', 'r') as file:
            self.data = json.load(file)

        self.keys = sorted(self.data.keys())
        self.slider_values = [round(self.data[key], 3) for key in self.keys]

    def create_sliders_and_entries(self):
        window_width = 600  # Width of the window
        label_width = max(len(key) for key in self.keys) + 5  # Calculate the width needed for the labels
        entry_width = 10  # Width for the entry boxes

        # Calculate the remaining width for the sliders
        slider_width = window_width - (label_width + entry_width + 20)  # 20 is an arbitrary padding value

        self.sliders = []
        self.entries = []

        for i, key in enumerate(self.keys):
            slider_frame = tk.Frame(self.scrollable_frame)
            slider_frame.grid(row=i, column=0, sticky='ew')

            label = tk.Label(slider_frame, text=key, width=label_width, anchor='w')
            label.grid(row=0, column=0, sticky="w")

            entry = tk.Entry(slider_frame, width=entry_width)
            entry.grid(row=0, column=1, sticky="ew")
            entry.insert(0, f"{self.slider_values[i]:.3f}")
            entry.bind('<Return>', self.update_from_entry(i))

            # Set a static width for the sliders
            slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, resolution=0.001, length=slider_width, command=self.update_from_slider(i))
            slider.set(self.slider_values[i])
            slider.grid(row=0, column=2, sticky="ew")

            self.sliders.append(slider)
            self.entries.append(entry)




    def on_frame_configure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_from_slider(self, index):
        def _update(value):
            value = round(float(value), 3)
            self.entries[index].delete(0, tk.END)
            self.entries[index].insert(0, f"{value:.3f}")
            self.adjust_weights(index, value)
        return _update

    def update_from_entry(self, index):
        def _update(event):
            value = round(float(self.entries[index].get()), 3)
            self.sliders[index].set(value)
            self.adjust_weights(index, value)
        return _update

    def adjust_weights(self, adjusted_index, new_value):
        self.slider_values[adjusted_index] = new_value
        other_total = 100 - new_value
        other_indices = [i for i in range(len(self.slider_values)) if i != adjusted_index]
        other_values = [self.slider_values[i] for i in other_indices]

        if sum(other_values) == 0:
            for i in other_indices:
                self.slider_values[i] = other_total / len(other_indices)
        else:
            scale_factor = other_total / sum(other_values)
            for i in other_indices:
                self.slider_values[i] = round(self.slider_values[i] * scale_factor, 3)

        for i, slider in enumerate(self.sliders):
            slider.set(self.slider_values[i])
            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, f"{self.slider_values[i]:.3f}")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def save_data(self):
        print('Saving Data...')
        for i, key in enumerate(self.keys):
            self.data[key] = self.slider_values[i]
        
        with open(f'./{self.data_template}/weight.json', 'w') as file:
            json.dump(self.data, file, indent=4)
        
        self.root.quit()

    def setup_buttons(self):
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(fill=tk.X, side="bottom", padx=5, pady=5)

        self.ok_button = tk.Button(self.buttons_frame, text="OK", command=self.save_data)
        self.ok_button.pack(side="left", expand=True, fill=tk.X)

        self.cancel_button = tk.Button(self.buttons_frame, text="Cancel", command=self.root.quit)
        self.cancel_button.pack(side="right", expand=True, fill=tk.X)


def main():
    try:
        parser = argparse.ArgumentParser(description='Weight Adjusting code')

        parser.add_argument('--data',
                    dest='data_template',
                    type=str,
                    required=True,
                    help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
        
        args = parser.parse_args()
        data_template = args.data_template

        check_file(f'./{data_template}/weight.json')

        root = tk.Tk()
        root.geometry("800x900")
        root.resizable(False,True)

        app = SliderWindow(root, data_template)
        root.mainloop()

    except Exception as E:
        print(E)
        return
    
if __name__ == "__main__":
    main()

