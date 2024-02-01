import tkinter as tk
import json

class SliderWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Weight Editor")

        # Set minimum and maximum window size
        self.root.minsize(400, 300)  # Minimum size
        self.root.maxsize(800, 600)  # Maximum size

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Configure the canvas to use the scrollbar
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Load slider values from JSON file
        with open('./kdd/weight.json', 'r') as file:
            self.data = json.load(file)

        self.keys = sorted(self.data.keys())
        self.slider_values = [self.data[key] for key in self.keys]
        self.sliders = []

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
            self.sliders.append(slider)

        # Pack the canvas and scrollbar into the main frame
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Buttons frame
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(fill=tk.X, side="bottom")

        # Span the OK button across the entire width of the buttons frame
        self.ok_button = tk.Button(self.buttons_frame, text="OK", command=self.save_data)
        self.ok_button.pack(fill=tk.X, padx=5, pady=5)

        # Bind window resize event
        self.root.bind("<Configure>", self.adjust_element_width)

        # Bind mouse scroll event for scrolling through sliders
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def adjust_element_width(self, event):
        width = self.root.winfo_width()
        self.ok_button.config(width=width)

    def save_data(self):
        with open('./kdd/weight.json', 'w') as file:
            json.dump(self.data, file, indent=4)
        self.root.quit()

    def scale_sliders(self, adjusted_index, new_value):
        other_total = 100 - new_value
        other_indices = [i for i in range(len(self.slider_values)) if i != adjusted_index]
        other_values = [self.slider_values[i] for i in other_indices]

        if sum(other_values) == 0:
            for i in other_indices:
                self.slider_values[i] = other_total / len(other_indices)
        else:
            scale_factor = other_total / sum(other_values)
            for i in other_indices:
                self.slider_values[i] *= scale_factor

    def update_sliders(self, adjusted_index):
        def _update(value):
            new_value = float(value)
            if self.slider_values[adjusted_index] != new_value:
                self.slider_values[adjusted_index] = new_value
                self.scale_sliders(adjusted_index, new_value)
                for i, slider in enumerate(self.sliders):
                    slider.set(self.slider_values[i])
        return _update

if __name__ == "__main__":
    root = tk.Tk()
    app = SliderWindow(root)
    root.mainloop()
