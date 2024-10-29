from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk  # You need to install the Python Imaging Library (PIL)

# from artificial_artwork._demo import create_algo_runner
from artificial_artwork._main import create_algo_runner
from artificial_artwork.image import convert_to_uint8

# runtime directory of this file script
my_dir: Path = Path(__file__).parent


# CONSTANTS
# Component assets for input at render time
IMAGE_COMP_ASSETS = {
    'content': {
        'load_button_text': "Select Content Image",
        'label_text': "Content Image:",
    },
    'style': {
        'load_button_text': "Select Style Image",
        'label_text': "Style Image:",
    },
}
# Default Content and Style images for quick test/nst-run
DEFAULT_CONTENT_IMAGE = my_dir / 'tests/data/canoe_water_w300-h225.jpg'
DEFAULT_STYLE_IMAGE = my_dir / 'tests/data/blue-red_w300-h225.jpg'

# width x height
WINDOW_GEOMETRY: str = '2600x1800'

# Content and Style Images rendering dimensions
INPUT_IMAGE_THUMBNAIL_SIZE = (200, 200)

# Generated Image rendering dimensions
GENERATED_IMAGE_THUMBNAIL_SIZE = (500, 500)


# Helpers Objects

img_type_2_path = {}

# Helper Functions
def _build_open_image_dialog_callback_v2(x, image_type: str, initial_file=None):
    def _open_file_dialog_v2():
        file_path = filedialog.askopenfilename(initialfile=initial_file)
        if file_path:
            image_label = x['image_label']
            image_pane = x['image_pane']

            img_type_2_path[image_type] = file_path

            image = Image.open(file_path)
            image.thumbnail(INPUT_IMAGE_THUMBNAIL_SIZE)  # Resize the image to fit in the pane
            photo = ImageTk.PhotoImage(image=image)

            image_pane.config(image=photo)
            image_pane.image = photo

            image_label.config(text=f'{IMAGE_COMP_ASSETS[image_type]["label_text"]} {file_path}')
            image_label.update_idletasks()
    return _open_file_dialog_v2


# MAIN

## GLOBAL STATE of the UI
stop_nst = False  # signal from user's interaction with UI
nst_running = False  # signal from backend's running state


images_components_data = {
    'content': dict(
        IMAGE_COMP_ASSETS['content'],
        # image_dialog_from_label=lambda label_obj: _build_open_image_dialog_callback(label_obj, 'content'),
        image_dialog=lambda x: _build_open_image_dialog_callback_v2(x, 'content', initial_file=DEFAULT_CONTENT_IMAGE),
    ),
    'style': dict(
        IMAGE_COMP_ASSETS['style'],
        # image_dialog_from_label=lambda label_obj: _build_open_image_dialog_callback(label_obj, 'style'),
        image_dialog=lambda x: _build_open_image_dialog_callback_v2(x, 'style', initial_file=DEFAULT_STYLE_IMAGE),
    ),
}

# Create the main window
root = tk.Tk()
root.title("Neural Style Transfer - Desktop")
# width x height
root.geometry("2600x1800")  # Larger window size

# Add a label to describe the purpose of the GUI
description_label = tk.Label(root, text="Select a file using the buttons below:")
description_label.pack(pady=10)  # Add padding


# CONTENT IMAGE UI/UX

# BUTTON -> Load Content Image
button1 = tk.Button(
    root,
    text=images_components_data['content']['load_button_text'],
    # command=lambda: images_components_data['content']['image_dialog_from_label'](content_image_label)(),
    command=lambda: images_components_data['content']['image_dialog']({
        'image_label': content_image_label,
        'image_pane': content_image_pane,
    })(),
)
button1.pack(pady=5)  # Add padding

# LABEL -> Show path of loaded Content Image
content_image_label = tk.Label(root, text=images_components_data['content']['label_text'])
content_image_label.pack()

# LABEL -> PANE to Render the Content Image
content_image_pane = tk.Label(root, width=0, height=0, bg="white")  # Set initial dimensions to 0
# content_image_pane = tk.Label(root, width=200, height=200, bg="white")
content_image_pane.pack()


# STYLE IMAGE UI/UX

# BUTTON -> Load Style Image
load_style_image_btn = tk.Button(
    root,
    text=images_components_data['style']['load_button_text'],
    # command=lambda: images_components_data['style']['image_dialog_from_label'](style_image_label)()
    command=lambda: images_components_data['style']['image_dialog']({
        'image_label': style_image_label,
        'image_pane': style_image_pane,
    })(),
)
load_style_image_btn.pack(pady=5)  # Add padding

# LABEL -> Show path of loaded Style Image
style_image_label = tk.Label(root, text=images_components_data['style']['label_text'])
style_image_label.pack()

# LABEL -> PANE to Render the Style Image
style_image_pane = tk.Label(root, width=0, height=0, bg="white")  # Set initial dimensions to 0
# style_image_pane = tk.Label(root, width=200, height=200, bg="white")
style_image_pane.pack()


# GENERATED IMAGE UI/UX

# Helper Update Callback
# def update_image_thread(progress, gen_image_pane, _iteration_count_label, fig, combined_subplot):
#     t = threading.Thread(
#         target=update_image,
#         args=(progress, gen_image_pane, _iteration_count_label, fig, combined_subplot)
#     )
#     t.start()

#### UPDATE UI based on BACKEND progress ####

# Function to update the GUI with the result from the backend task
# def update_image(progress, gen_image_pane, _iteration_count_label, fig, combined_subplot):
def update_image_thread(progress, gen_image_pane, _iteration_count_label, fig, combined_subplot):
    numpy_image_array = progress.state.matrix
    current_iteration_count: int = progress.state.metrics['iterations']

    # if we have shape of form (1, Width, Height, Number_of_Color_Channels)
    if numpy_image_array.ndim == 4 and numpy_image_array.shape[0] == 1:
        # reshape to (Width, Height, Number_of_Color_Channels)
        matrix = np.reshape(numpy_image_array, tuple(numpy_image_array.shape[1:]))

    if str(matrix.dtype) != 'uint8':
        matrix = convert_to_uint8(matrix)

    image = Image.fromarray(matrix)

    # Resize the image to fit in the pane
    image.thumbnail(GENERATED_IMAGE_THUMBNAIL_SIZE)
    # Convert the image to PhotoImage
    photo = ImageTk.PhotoImage(image=image)
    # Update the image label with the new image
    gen_image_pane.config(image=photo)
    gen_image_pane.image = photo

    _iteration_count_label.config(text=f'Iteration Count: {current_iteration_count}')

    if 'cost' in progress.state.metrics:  # backend has evaluated the costs into scalars (floats)
        # Update metrics
        total_cost_values.append(progress.state.metrics['cost'])
        style_cost_values.append(progress.state.metrics['style-cost-weighted'])
        content_cost_values.append(progress.state.metrics['content-cost-weighted'])
        iteration_values.append(current_iteration_count)

        # Update the graph
        update_chart(
            iteration_values,
            total_cost_values,
            style_cost_values,
            content_cost_values,
            combined_subplot
        )

################

# LABEL -> Text to display above Live Updated Generated Image
generated_image_label = tk.Label(root, text="Generated Image:")
generated_image_label.pack(pady=10)

# LABEL -> Live Display of Generated Image ! (this will be updated during the learning loop)
generated_image_pane = tk.Label(root, width=0, height=0, bg="white")  # Set initial dimensions to 0
generated_image_pane.pack(pady=5)

# ITERATION COUNT UI/UX
# LABEL -> Iteration Count Live Update
iteration_count_label = tk.Label(root, text="Iteration Count:")
iteration_count_label.pack(pady=5)


# RUN NST ALGORITHM UI/UX

# Helper Run Functions

# Run NST Computations in a non-blocking way

def run_nst(fig, combined_subplot):
    # Run tf.compat.v1.reset_default_graph()
    # and tf.compat.v1.disable_eager_execution()
    # Initialize Session as tf.compat.v1.InteractiveSession()
    backend_object = create_algo_runner(
        iterations=100,  # NB of Times to pass Image through the Network
        output_folder='gui-output-folder',  # Output Folder to store gen img snapshots
        noisy_ratio=0.6,
    )
    observer = type('Observer', (), {
        'update': lambda progress: update_image_thread(
            progress,
            generated_image_pane,
            iteration_count_label,
            fig,  # Pass the Figure to the update function
            combined_subplot,  # Pass the combined subplot to the update function
        ),
        # 'update': lambda progress: update_image_thread(progress, generated_image_pane, iteration_count_label),
    })
    backend_object['subscribe'](observer)

    content_image_path = img_type_2_path['content']
    style_image_path = img_type_2_path['style']

    if content_image_path and style_image_path:
        backend_object['run'](
            content_image_path,
            style_image_path,
        )

# Define Tread to run the NST Algorithm
def start_nst_thread():
    fig, combined_subplot = initialize_graph(root)
    nst_thread = threading.Thread(target=run_nst, args=(fig, combined_subplot))
    nst_thread.daemon = True  # Set as a daemon thread to exit when the main program exits
    nst_thread.start()


# BUTTON -> Run NST Algorithm on press
run_nst_btn = tk.Button(
    root,
    text="Run NST Algorithm",
    command=start_nst_thread,
)

run_nst_btn.pack(pady=5)  # Add padding


# PLOTTING

total_cost_values = []
style_cost_values = []
content_cost_values = []
iteration_values = []


# Helper Functions
# Initialize Matplotlib figure and subplot
def initialize_graph(root):
    fig, combined_subplot = plt.subplots(figsize=(8, 6))
    combined_subplot.set_title('Metrics Over Iterations')
    combined_subplot.set_xlabel('Iterations')
    combined_subplot.set_ylabel('Metric Values')
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    return fig, combined_subplot


# Update Matplotlib chart with metrics data
def update_chart(_iteration_values, _total_cost_values, _style_cost_values, _content_cost_values, _combined_subplot):
    _combined_subplot.clear()
    _combined_subplot.plot(_iteration_values, _total_cost_values, label='Total Cost', marker='o')
    _combined_subplot.plot(_iteration_values, _style_cost_values, label='Weighted Style Cost', marker='s')
    _combined_subplot.plot(_iteration_values, _content_cost_values, label='Weighted Content Cost', marker='x')
    _combined_subplot.set_title('Metrics Over Iterations')
    _combined_subplot.set_xlabel('Iterations')
    _combined_subplot.set_ylabel('Metric Values')
    _combined_subplot.legend()

    _combined_subplot.figure.canvas.draw()


# TKINTER MAIN LOOP
root.mainloop()
