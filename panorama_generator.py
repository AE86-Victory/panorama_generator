'''
Author: Hang SHENG
Student ID: 20321587
Email: scyhs3@nottingham.edu.cn
'''

import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from threading import Thread
from PIL import Image, ImageTk, ImageEnhance
import time


def extract_frames(video_path):
    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps / float(selected_skip_frames.get()))
    color_frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if selected_mode.get() == 'Night':
            # Increasing brightness and reduce noises for each frame
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=20)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        if idx % skip_frames == 0:
            color_frames.append(frame)
        idx += 1
    cap.release()
    return color_frames


def create_panorama(color_frames):
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(color_frames)
    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        return None
        # Solve black borders
    pano = crop_black_borders(pano)
    pano = inpaint_black_edges(pano)

    # Using pillow to enhance image to get better quality
    pano_pil = Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))  # Transfer to pil
    pano_pil = enhance_image_with_pil(pano_pil)
    pano = cv2.cvtColor(np.array(pano_pil), cv2.COLOR_RGB2BGR)  # Transfer to openCV

    return pano


def enhance_image_with_pil(pil_image):

    # Saturation enhancement
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.3)

    # Contrast enhancement
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.1)

    # Sharpness enhancement
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)

    return pil_image


def crop_black_borders(image):
    # Converting images to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Finding the non-black area
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_image = image.copy()
    # cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        best_rect = (x, y, w, h)
    else:
        best_rect = None

    # Crop
    if best_rect:
        #cv2.rectangle(contours_image, (best_rect[0], best_rect[1]), (best_rect[0] + best_rect[2], best_rect[1] + best_rect[3]), (255, 0, 0), 5)
        final_cropped = image[best_rect[1]:best_rect[1] + best_rect[3],
                        best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        final_cropped = image

    return final_cropped


def inpaint_black_edges(image):
    # Find black border and inpainting it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return inpainted_image


def load_image_into_label(image_path, label):
    try:
        # Using Pillow to load image
        pil_image = Image.open(image_path)
        # Convert Pillow image objects to Tkinter-compatible PhotoImage objects.
        tk_image = ImageTk.PhotoImage(pil_image)

        # Displaying images on labels
        label.config(image=tk_image)
        label.image = tk_image
    except Exception as e:
        print(f"Load failure: {e}")


def save_panorama(video_path, panorama):
    # Set the generated panorama filename to be the same as the video filename
    video_name = os.path.basename(video_path)
    panorama_name = os.path.splitext(video_name)[0] + "_panorama.jpg"
    cv2.imwrite(panorama_name, panorama)
    return panorama_name


def create_gui():
    # Main window
    window = tk.Tk()
    window.title("Panorama Generator")
    window.geometry('800x600')

    global image_label, current_image
    image_label = tk.Label(window)
    image_label.pack(fill=tk.BOTH, expand=True)
    current_image = None

    def on_window_resize(event):
        if current_image:
            # Adjust the size of image and show it in label
            resize_and_show_image(event.width, event.height)

    def resize_and_show_image(width, height):
        # Adjust the size of image
        img_width, img_height = current_image.size
        ratio = min(width / img_width, height / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))

        if new_size[0] <= 0 or new_size[1] <= 0:
            return  # Avoiding program lag

        resized_image = current_image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)
        image_label.config(image=photo)
        image_label.image = photo

    window.bind('<Configure>', on_window_resize)  # Binding window resize events

    # Load video and extract frames
    def load_video():
        video_path = filedialog.askopenfilename(title="Choose one video",
                                                filetypes=[("MP4 files", "*.mp4"), ("MOV files", "*.mov"),
                                                           ("All files", "*.*")])
        if video_path:
            status_label.config(text="Extracting frames")
            # Running long operations in background threads
            Thread(target=extract_and_stitch, args=(video_path,)).start()

    def extract_and_stitch(video_path):
        start_time = time.time()
        color_frames = extract_frames(video_path)
        panorama = create_panorama(color_frames)
        if panorama is not None:
            end_time = time.time()
            show_image(panorama)
            panorama_name = save_panorama(video_path, panorama)
            status_label.config(
                text=f"Success！Saved as {panorama_name}. \nTime taken: {end_time - start_time:.2f} seconds.")
        else:
            status_label.config(text="Failure")

    def show_image(image):
        global current_image
        # Transfer to PIL and show in the Tkinder
        current_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resize_and_show_image(window.winfo_width(), window.winfo_height())

    # Create a Frame to hold the radio buttons
    global selected_mode
    mode_frame = tk.Frame(window, bg='lightgray', pady=10, padx=10)
    mode_frame.pack(pady=5, padx=10, anchor='center')  # 将 Frame 居中
    mode_label = tk.Label(mode_frame, text="Select Video Type:", bg='lightgray')
    mode_label.pack(anchor="n")
    selected_mode = tk.StringVar(value="Day")  # Default is daytime mode
    tk.Radiobutton(mode_frame, text="Day time Video", variable=selected_mode, value="Day").pack(side=tk.LEFT)
    tk.Radiobutton(mode_frame, text="Night time Video", variable=selected_mode, value="Night").pack(side=tk.LEFT)

    def update_label(event):
        rate_label.config(text=f"Current Rate: {selected_skip_frames.get():.1f} frame(s)/sec")

    # Add frame skip selection radio button

    global selected_skip_frames
    skip_frame_frame = tk.Frame(window, bg='lightgray', pady=10, padx=10)
    skip_frame_frame.pack(pady=5, padx=10, anchor='center')
    selected_skip_frames = tk.DoubleVar(value=1.0)  # 默认跳过帧速率为每秒1帧
    skip_slider = tk.Scale(window, from_=1.0, to=3.0, variable=selected_skip_frames, orient='horizontal',
                           command=update_label)
    skip_slider.pack(pady=5)

    rate_label = tk.Label(window, text=f"Current Rate: {selected_skip_frames.get():.1f} frame(s)/sec")
    rate_label.pack()

    # Loading button
    load_video_btn = tk.Button(window, text="Loading your video", command=load_video)
    load_video_btn.pack(pady=10)

    # status label
    status_label = tk.Label(window, text="Please loading your video", font=("Arial", 12))
    status_label.pack(pady=10)

    window.mainloop()


if __name__ == '__main__':
    create_gui()
