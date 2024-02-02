import tkinter as tk
import json

class SliderWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Weight Editor")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and a scrollbar for vertical scrolling
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
    
        self.root.minsize(400, 300)  # Minimum size
        self.root.maxsize(800, 600)  # Maximum size
        
        # Mouse scroll event for the canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Load slider values from JSON file
        with open('./kdd/weight.json', 'r') as file:
            self.data = json.load(file)

        self.keys = sorted(self.data.keys())
        self.slider_values = [self.data[key] for key in self.keys]

        label_width = max(len(key) for key in self.keys) + 5

        for i, key in enumerate(self.keys):
            slider_frame = tk.Frame(self.scrollable_frame)
            label = tk.Label(slider_frame, text=key, width=label_width, anchor='w')
            label.grid(row=0, column=0, sticky="w")
            slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_sliders(i))
            slider.set(self.slider_values[i])
            slider.grid(row=0, column=1, sticky="ew")
            slider_frame.columnconfigure(1, weight=1)
            slider_frame.pack(fill=tk.X, expand=True)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Buttons frame
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(fill=tk.X, side="bottom", padx=5, pady=5)

        self.ok_button = tk.Button(self.buttons_frame, text="OK", command=self.save_data)
        self.ok_button.pack(side="left", expand=True, fill=tk.X, padx=5)

        self.cancel_button = tk.Button(self.buttons_frame, text="Cancel", command=self.root.quit)
        self.cancel_button.pack(side="right", expand=True, fill=tk.X, padx=5)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def save_data(self):
        with open('./kdd/weight.json', 'w') as file:
            json.dump(self.data, file, indent=4)
        self.root.quit()

    def update_sliders(self, adjusted_index):
        def _update(value):
            new_value = float(value)
            self.data[self.keys[adjusted_index]] = new_value  # Update the data dictionary
            # Scale the other sliders accordingly
            self.scale_sliders(adjusted_index, new_value)
            # Update the slider positions
            for i, key in enumerate(self.keys):
                self.scrollable_frame.winfo_children()[i].winfo_children()[1].set(self.data[key])
        return _update

    def scale_sliders(self, adjusted_index, new_value):
        other_total = 100 - new_value
        other_indices = [i for i in range(len(self.slider_values)) if i != adjusted_index]
        other_values = [self.data[self.keys[i]] for i in other_indices]

        if sum(other_values) == 0:
            for i in other_indices:
                self.data[self.keys[i]] = other_total / len(other_indices)
        else:
            scale_factor = other_total / sum(other_values)
            for i in other_indices:
                self.data[self.keys[i]] *= scale_factor

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")  # Initial size of the window
    app = SliderWindow(root)
    root.mainloop()

    
