import tkinter as tk
from tkinter import ttk
import json
import argparse

import subprocess
import sys

from module.file_op import check_file
from module.testing_module import *
class SliderWindow:
    def __init__(self, root, data_template, threshold):
        self.root = root
        self.root.title("Weight Editor")
        
        self.data_template = data_template
        self.threshold = threshold
        
        self.setup_main_frame()
        self.setup_threshold_label()
        self.setup_scrollable_area()
        self.load_data()
        self.create_sliders_and_entries()
        self.setup_buttons()
        
    def setup_main_frame(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def setup_threshold_label(self):
        self.threshold_frame = tk.Frame(self.main_frame)
        self.threshold_frame.pack(fill=tk.X, pady=10)

        # Inner frame to hold the label and entry, centered within threshold_frame
        inner_frame = tk.Frame(self.threshold_frame)
        inner_frame.pack(anchor='center')

        tk.Label(inner_frame, text="Threshold:", font=(12)).pack(side=tk.LEFT)
        
        val = (self.root.register(self.validate_threshold), '%P')  # Validation command
        self.threshold_entry = tk.Entry(inner_frame, font=(12), validate="key", validatecommand=val)
        self.threshold_entry.pack(side=tk.LEFT, padx=10)
        self.threshold_entry.insert(0, str(self.threshold))
            
    def validate_threshold(self, P):
        if P == "" or (P.replace('.', '', 1).isdigit() and 0 <= float(P) <= 100):
            return True
        else:
            self.root.bell()
            return False
    
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

        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.scrollable_window, width=canvas_width - 20)


    def on_frame_configure(self, event=None):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    
    def load_data(self):
        with open(f'./{self.data_template}/weight.json', 'r') as file:
            self.data = json.load(file)

        self.keys = sorted(self.data.keys())
        self.slider_values = [round(self.data[key], 3) for key in self.keys]

    def create_sliders_and_entries(self):
        window_width = 600
        label_width = max(len(key) for key in self.keys) + 5
        entry_width = 10

        slider_width = window_width - (label_width + entry_width + 5)

        self.sliders = []
        self.entries = []
        self.checkbuttons = []
        self.checkbutton_states = []
        
        label_frame = tk.Frame(self.scrollable_frame)
        label_frame.grid(row=0, column=0, sticky='ew')

        tk.Label(label_frame, text='Attack', width=label_width, anchor='w').grid(row=0, column=0)
        tk.Label(label_frame, text='Hold', width=10, anchor='w').grid(row=0, column=1)
        tk.Label(label_frame, text='Weight', width=entry_width, anchor='w').grid(row=0, column=2, padx=20)


        for i, key in enumerate(self.keys):
            slider_frame = tk.Frame(self.scrollable_frame)
            slider_frame.grid(row=i+1, column=0, sticky='ew')  
            
            label = tk.Label(slider_frame, text=key, width=label_width, anchor='w')
            label.grid(row=0, column=0, sticky="w")

            var = tk.IntVar()
            check = tk.Checkbutton(slider_frame, variable=var, command=lambda i=i: self.toggle_slider_state(i))
            check.grid(row=0, column=1, sticky="w")
            self.checkbuttons.append(check)
            self.checkbutton_states.append(var)

            entry = tk.Entry(slider_frame, width=entry_width)
            entry.grid(row=0, column=2, sticky="ew")
            entry.insert(0, f"{self.slider_values[i]:.3f}")
            entry.bind('<Return>', self.update_from_entry(i))
            
            slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, resolution=0.001, length=slider_width, command=self.update_from_slider(i))
            
            slider.set(self.slider_values[i])
            slider.grid(row=0, column=3, sticky="ew")  

            self.sliders.append(slider)
            self.entries.append(entry)
            
    def get_max_value(self, adjusted_index):
        total_checked_value = sum(self.slider_values[i] for i, var in enumerate(self.checkbutton_states) if var.get() == 1)

        total_unchecked_value = sum(self.slider_values[i] for i, var in enumerate(self.checkbutton_states) if var.get() == 0 and i != adjusted_index)

        max_value = 100 - (total_checked_value + total_unchecked_value)
        return max(0, min(max_value, 100))

    def toggle_slider_state(self, index):
        state = 'disabled' if self.checkbutton_states[index].get() == 1 else 'normal'
        self.sliders[index].config(state=state)
        self.entries[index].config(state=state)

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
    
    def calculate_max_value(self, adjusted_index, new_value, locked_indices):
        total_locked_value = sum(self.slider_values[i] for i in locked_indices)

        remaining_value = 100 - total_locked_value - new_value

        if remaining_value < 0:
            new_value += remaining_value

        return max(0, new_value)
    
    def adjust_weights(self, adjusted_index, new_value):
        locked_indices = [i for i, var in enumerate(self.checkbutton_states) if var.get() == 1]

        if adjusted_index in locked_indices:
            return

        max_value = self.calculate_max_value(adjusted_index, float(new_value), locked_indices)

        self.slider_values[adjusted_index] = max_value
        self.sliders[adjusted_index].set(max_value)
        self.entries[adjusted_index].delete(0, tk.END)
        self.entries[adjusted_index].insert(0, f"{max_value:.3f}")

        self.redistribute_weights(adjusted_index, locked_indices)

    def redistribute_weights(self, adjusted_index, locked_indices):
        total_allocation = 100 - sum(self.slider_values[i] for i in locked_indices)
        remaining_allocation = total_allocation - self.slider_values[adjusted_index]

        non_locked_indices = [i for i in range(len(self.slider_values)) if i not in locked_indices and i != adjusted_index]
        total_current_allocation = sum(self.slider_values[i] for i in non_locked_indices)

        if total_current_allocation == 0:
            for i in non_locked_indices:
                self.slider_values[i] = remaining_allocation / len(non_locked_indices)
        else:
            scale_factor = remaining_allocation / total_current_allocation
            for i in non_locked_indices:
                new_value = self.slider_values[i] * scale_factor
                self.slider_values[i] = max(new_value, 0)

        for i in non_locked_indices:
            self.sliders[i].set(self.slider_values[i])
            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, f"{self.slider_values[i]:.3f}")
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def save_data(self):
        print('Saving Data...')
        try:
            new_threshold = float(self.threshold_entry.get())
            # Validate the new threshold value
            if 0 <= new_threshold <= 100:
                threshold_data = {'threshold': new_threshold}
                with open(f'./{self.data_template}/Result/threshold.json', 'w') as file:
                    json.dump(threshold_data, file, indent=4)
                print('Threshold saved successfully.')
            else:
                print("Invalid threshold value. Please enter a number between 0 and 100.")
                self.root.bell()  # Alert the user to the error
        except ValueError:
            print("Invalid input for threshold. Please enter a valid number.")
            self.root.bell()  
            return  # Exit the function without attempting to save other data
        for i, key in enumerate(self.keys):
            self.data[key] = self.slider_values[i]
        
        self.data = process_Largest_remainder_method(self.data, 3)
        
        with open(f'./{self.data_template}/weight.json', 'w') as file:
            json.dump(self.data, file, indent=4)
            
        new_threshold = float(self.threshold_entry.get())
        threshold_data = {'threshold': new_threshold}
        
        with open(f'./{self.data_template}/Result/threshold.json', 'w') as file:
            json.dump(threshold_data, file, indent=4)

        print('Threshold and data saved successfully.')
        self.root.quit()

    def setup_buttons(self):
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(fill=tk.X, side="bottom", padx=5, pady=5)

        self.ok_button = tk.Button(self.buttons_frame, text="OK", command=self.save_data)
        self.ok_button.pack(side="left", expand=True, fill=tk.X)
        
        self.reset_button = tk.Button(self.buttons_frame, text="Reset Weights", command=self.reset_weights)
        self.reset_button.pack(side="right", expand=True, fill=tk.X)
        
        self.cancel_button = tk.Button(self.buttons_frame, text="Cancel", command=self.root.quit)
        self.cancel_button.pack(side="right", expand=True, fill=tk.X)
        
    def reset_weights(self):
        # Run the weight_reset.py
        subprocess.run([sys.executable, './weight_reset.py', '--data', self.data_template])

        # Close window
        self.root.destroy()

        # Restart
        subprocess.run([sys.executable, 'weight.py', '--data', self.data_template])

def main():
    try:
        weight_decimal = 3
        parser = argparse.ArgumentParser(description='Weight Adjusting code')

        parser.add_argument('--data',
                    dest='data_template',
                    type=str,
                    required=True,
                    help='The data struture. The default data structures is cic (CICIDS2017) and kdd (NSL-KDD). (*Require)')
        
        args = parser.parse_args()
        data_template = args.data_template

        weight_path = f'./{data_template}/weight.json'
        threshold_path = f'./{data_template}/Result/threshold.json'
        
        isweight = check_and_return_file(weight_path)
        isthresh = check_and_return_file(threshold_path)
        
        if isweight == False or isthresh == False:
            subprocess.run([sys.executable, 'weight.py', '--data', data_template])
        
        with open(threshold_path) as jsonfile:
            threshold = json.load(jsonfile).get('threshold')
        
        root = tk.Tk()
        root.geometry("100x900")
        root.resizable(False,True)
        root.maxsize(900, 1440)
        root.minsize(900, 600)
        app = SliderWindow(root, data_template, threshold)
        root.mainloop()

    except Exception as E:
        print(E)
        return
    
if __name__ == "__main__":
    main()

