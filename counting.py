import os
import argparse
import numpy as np
# import cv2
from PIL import Image
from datetime import datetime
import argparse
import time
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import tkinter as tk
from tkinter import filedialog

def analyse_images(images_path, model, labels, threshold, out):
    print("Starting...")
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    arrayVehicles = [0] * 4

    for filename in os.listdir(images_path):
        f = os.path.join(images_path, filename)
        image = Image.open(f)
        square_img = Image.new('RGB', (image.width, image.width), (255, 255, 255))
        square_img.paste(image, (0, 0))
        image = square_img
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        interpreter.invoke()
        objs = detect.get_objects(interpreter, threshold, scale)
        for obj in objs:
            if obj.id == 0:
                # car
                arrayVehicles[0] += 1
            elif obj.id == 1:
                # motorcycle
                arrayVehicles[1] += 1
            elif obj.id == 2:
                # truck
                arrayVehicles[2] += 1
            elif obj.id == 3:
                # van
                arrayVehicles[3] += 1
    print(arrayVehicles)

    print("end")


if __name__ == '__main__':

    model = "/home/elizaveta/Desktop/lab/checkpoint_saved_14118.tflite"
    labels = "labels.txt"
    out = "/home/elizaveta/Desktop/lab/results"
    threshold = 0.5

    root = tk.Tk()
    root.directory = filedialog.askdirectory()
    images_path = root.directory

    analyse_images(images_path, model, labels, threshold, out)
