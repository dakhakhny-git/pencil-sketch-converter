import cv2
import numpy as np
import os
from sketch import pencil_sketch, edge_sketch

def process_single(input_path, output_path, mode="classic",
                    blur=21, strength=256):

    img = cv2.imread(input_path)

    if img is None:
        print(f" Error reading: {input_path}")
        return

    if blur % 2 == 0:
        blur += 1

    if mode == "classic":
        result = pencil_sketch(img, blur, strength)

    elif mode == "edge":
        sketch = pencil_sketch(img, blur, strength)
        edges = edge_sketch(img)
        result = cv2.bitwise_and(sketch, edges)

    else:
        print(" Invalid mode")
        return

    cv2.imwrite(output_path, result)
    print(f" Saved: {output_path}")




def process_folder(input_folder="images", output_folder="outputs", mode="classic", blur=21, strength=256):

    if not os.path.exists(input_folder):
        print(" Input folder not found!")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, "sketch_" + filename)

            process_single(input_path, output_path, mode, blur, strength)



if __name__ == "__main__":

    print("Pencil Sketch Converter")

    # User inputs
    mode = input("Choose mode (classic / edge): ").strip().lower()
    if mode not in ["classic", "edge"]:
        print("Invalid mode. Defaulting to classic.")
        mode = "classic"

    try:
        blur = int(input("Enter blur size (odd number, e.g. 21): "))
        strength = int(input("Enter sketch strength (e.g. 256): "))
    except:
        print("Invalid input. Using default values.")
        blur = 21
        strength = 256

    process_folder(
        input_folder="images",
        output_folder="outputs",
        mode=mode,
        blur=blur,
        strength=strength
    )