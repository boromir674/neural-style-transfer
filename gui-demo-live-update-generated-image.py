import tkinter as tk
from tkinter import filedialog
import threading
from PIL import Image, ImageTk  # You need to install the Python Imaging Library (PIL)

# from artificial_artwork._demo import create_algo_runner
from artificial_artwork._main import create_algo_runner
from artificial_artwork.image import convert_to_uint8


# CONSTANTS
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

# width x height
WINDOW_GEOMETRY: str = '2600x1800'

# Content and Style Images rendering dimensions
INPUT_IMAGE_THUMBNAIL_SIZE = (200, 200)

# Generated Image rendering dimensions
GENERATED_IMAGE_THUMBNAIL_SIZE = (500, 500)


# Helpers Objects

img_type_2_path = {}

# Helper Functions
def _build_open_image_dialog_callback(image_file_label, image_type: str):
    def _open_image_dialog():
        file_path = filedialog.askopenfilename()
        if file_path:
            img_type_2_path[image_type] = file_path
            image_file_label.config(text=f'{IMAGE_COMP_ASSETS[image_type]["label_text"]} {file_path}')
    return _open_image_dialog


def _build_open_image_dialog_callback_v2(x, image_type: str):
    def _open_file_dialog_v2():
        file_path = filedialog.askopenfilename()
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

images_components_data = {
    'content': dict(
        IMAGE_COMP_ASSETS['content'],
        image_dialog_from_label=lambda label_obj: _build_open_image_dialog_callback(label_obj, 'content'),
        image_dialog=lambda x: _build_open_image_dialog_callback_v2(x, 'content'),
    ),
    'style': dict(
        IMAGE_COMP_ASSETS['style'],
        image_dialog_from_label=lambda label_obj: _build_open_image_dialog_callback(label_obj, 'style'),
        image_dialog=lambda x: _build_open_image_dialog_callback_v2(x, 'style'),
    ),
}


# def open_file_dialog():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         file_label.config(text=f'Selected file: {file_path}')

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
def update_image_thread(progress, gen_image_pane, _iteration_count_label):
    t = threading.Thread(
        target=update_image,
        args=(progress, gen_image_pane, _iteration_count_label)
    )
    t.start()

# Function to update the GUI with the result from the backend task
def update_image(progress, gen_image_pane, _iteration_count_label):
    numpy_image_array = progress.state.matrix
    current_iteration_count: int = progress.state.metrics['iterations']

    # if we have shape of form (1, Width, Height, Number_of_Color_Channels)
    if numpy_image_array.ndim == 4 and numpy_image_array.shape[0] == 1:
        import numpy as np
        # reshape to (Width, Height, Number_of_Color_Channels)
        matrix = np.reshape(numpy_image_array, tuple(numpy_image_array.shape[1:]))

    if str(matrix.dtype) != 'uint8':
        matrix = convert_to_uint8(matrix)

    ## Prod code: broken
    # convert numpy array to PIL image
    # image = Image.fromarray(numpy_image_array)
    ##

    image = Image.fromarray(matrix)

    # Resize the image to fit in the pane
    image.thumbnail(GENERATED_IMAGE_THUMBNAIL_SIZE)
    # Convert the image to PhotoImage
    photo = ImageTk.PhotoImage(image=image)
    # Update the image label with the new image
    gen_image_pane.config(image=photo)
    gen_image_pane.image = photo

    _iteration_count_label.config(text=f'Iteration Count: {current_iteration_count}')

backend_object = create_algo_runner(
    iterations=100,  # NB of Times to pass Image through the Network
    output_folder='gui-output-folder',  # Output Folder to store gen img snapshots
    noisy_ratio=0.6,
)
# {
#     'algorithm_runner': algorithm_runner,
#     'run': lambda: algorithm_runner.run(algorithm, model_design),
#     'subscribe': lambda observer: algorithm_runner.progress_subject.add(observer),
# }

observer = type('Observer', (), {
    # 'update': lambda progress: update_image(progress, generated_image_pane, iteration_count_label),
    'update': lambda progress: update_image_thread(progress, generated_image_pane, iteration_count_label),
})
backend_object['subscribe'](observer)

# Pane for displaying generated image (this will be updated during the learning loop)
generated_image_label = tk.Label(root, text="Generated Image:")
generated_image_label.pack(pady=10)

generated_image_pane = tk.Label(root, width=0, height=0, bg="white")  # Set initial dimensions to 0
# generated_image_pane = tk.Label(root, width=600, height=600, bg="white")
generated_image_pane.pack(pady=5)

# ITERATION COUNT UI/UX
iteration_count_label = tk.Label(root, text="Iteration Count:")
iteration_count_label.pack(pady=5)


# RUN NST ALGORITHM UI/UX

# Helper Run Functions

# Run Computations
def run_nst():
    backend_object = create_algo_runner(
        iterations=100,  # NB of Times to pass Image through the Network
        output_folder='gui-output-folder',  # Output Folder to store gen img snapshots
        noisy_ratio=0.6,
    )
    observer = type('Observer', (), {
        'update': lambda progress: update_image(progress, generated_image_pane, iteration_count_label),
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

import concurrent.futures
# Function to run the backend task in a separate thread
def start_backend_task():
    
    # # Create your backend object using create_algo_runner
    # backend_object = create_algo_runner(iterations=100, output_folder='gui-output-folder', noisy_ratio=0.6)
    
    # # Define an observer object to handle progress updates, by updating the UI
    # observer = type('Observer', (), {
    #     'update': lambda progress: update_image(progress, generated_image_pane, iteration_count_label),
    #     # 'update': lambda progress: update_image_thread(progress, generated_image_pane, iteration_count_label),
    # })   

    # # Subscribe the observer to the backend object's progress
    # backend_object['subscribe'](observer)

    content_image_path = img_type_2_path['content']
    style_image_path = img_type_2_path['style']

    if content_image_path and style_image_path:
        def _run():
            backend_object['run'](
                content_image_path,
                style_image_path,
            )
        # Create a thread to run the backend task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run)


# Function to execute run_nst in a separate thread
# def start_nst_thread():
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(run_nst)
        # You can optionally add callbacks for handling the results

# Threaded Run Computations
# def start_nst_thread():
#     import tensorflow as tf

#     # Create a new TensorFlow graph and session in the new thread
#     # with tf.Graph().as_default(), tf.Session() as sess:
#         # Define TensorFlow operations
#         # ...

#         # Enqueue TensorFlow operations to be executed in the new thread
#     coord = tf.train.Coordinator()
#     enqueue_thread = tf.train.QueueRunner(tf.train.string_input_producer(["dummy_data"]))
#     threads = enqueue_thread.create_threads(sess, coord=coord, start=True)

#     # Start the TensorFlow operations within the new thread
#     run_nst()

def start_nst_thread():
    nst_thread = threading.Thread(target=run_nst)
    nst_thread.daemon = True  # Set as a daemon thread to exit when the main program exits
    nst_thread.start()

# BUTTON -> Run NST Algorithm on press
# run_nst_btn = tk.Button(
#     root,
#     text="Run NST Algorithm",
#     command=start_nst_thread,  # Start the thread when the button is pressed
# )
run_nst_btn = tk.Button(
    root,
    text="Run NST Algorithm",
    command=start_nst_thread,
)

run_nst_btn.pack(pady=5)  # Add padding


# Add a label to display the selected file
# file_label = tk.Label(root, text="", wraplength=300)  # Wrap text for better display
# file_label.pack(pady=10)  # Add padding


# style_image_pane = tk.Label(root, width=200, height=200, bg="white")
# style_image_pane.pack()

root.mainloop()
