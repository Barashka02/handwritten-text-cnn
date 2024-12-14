import torch
import numpy as np
import imageio
import scipy.ndimage as ndi
from PIL import Image
from models.cnn import CNN
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, disk, closing, opening, erosion
import matplotlib.pyplot as plt
import math


import matplotlib.patches as patches

def extract_characters(image_path, model_path='emnist_letters_cnn.pth', min_area=3000, pad=5, space_threshold=1.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN(num_classes=26).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    thresh = threshold_otsu(img)
    binary = img < thresh

    selem = disk(2)
    # open to remove separate any letters that are touching
    binary_dilated = opening(binary, selem)

    # find all the letters
    labeled, num_features = ndi.label(binary_dilated)
    objects = ndi.find_objects(labeled)

    # filter objects based on area (to remove noise if present)
    filtered_objects = []
    for i, slc in enumerate(objects):
        if slc is not None:
            component = binary_dilated[slc]
            area = np.sum(component)
            if area >= min_area:
                filtered_objects.append(slc)


    # sort the object to be in the order of appearance
    filtered_objects = sorted(filtered_objects, key=lambda slc: slc[1].start)


    # Plot the original image with bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img, cmap='gray')
    ax.set_title('Detected Characters with Bounding Boxes')

    chars_and_positions = []

    for i, slc in enumerate(filtered_objects):
        char_img = binary[slc]

        # Pad to square
        padding = pad
        h, w = char_img.shape
        max_dim = max(h, w) + 2 * padding
        padded = np.zeros((max_dim, max_dim), dtype=char_img.dtype)
        offset_y = (max_dim - h) // 2
        offset_x = (max_dim - w) // 2
        padded[offset_y:offset_y + h, offset_x:offset_x + w] = char_img

        # Resize to 28x28
        pil_img = Image.fromarray((padded * 255).astype(np.uint8))
        pil_img = pil_img.resize((28, 28), Image.BILINEAR)

        # Predict the character
        char_tensor = torch.from_numpy(np.array(pil_img)).float().unsqueeze(0).unsqueeze(0)
        char_tensor = (char_tensor / 255.0 - 0.1307) / 0.3081
        char_tensor = char_tensor.to(device)

        with torch.no_grad():
            output = model(char_tensor)
            pred = output.argmax(dim=1).item()

        char = chr(pred + ord('a'))

        # Record character and its position
        start_col = slc[1].start
        chars_and_positions.append((char, start_col))

    #     # Draw the bounding box on the original image
        y1, x1 = slc[0].start, slc[1].start
        y2, x2 = slc[0].stop, slc[1].stop
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, char, color='red', fontsize=12, fontweight='bold')

    plt.show()


    # Calculate the gaps between consecutive characters
    gaps = []
    for i in range(1, len(chars_and_positions)):
        current_start = chars_and_positions[i][1]
        previous_start = chars_and_positions[i - 1][1]
        gap = current_start - previous_start
        gaps.append(gap)

    # Calculate the average gap
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
    else:
        avg_gap = 0  # If only one character, no gaps

    # Set a threshold for space det
    # ection (adjust the multiplier as needed)
    space_threshold = avg_gap * space_threshold # You can experiment with 1.5, 2.0, etc.

    # Build the output string, inserting spaces where appropriate
    output_chars = [chars_and_positions[0][0]]  # Start with the first character

    for i in range(1, len(chars_and_positions)):
        current_char, current_start = chars_and_positions[i]
        previous_char, previous_start = chars_and_positions[i - 1]
        gap = current_start - previous_start
        if gap > space_threshold and gap:
            output_chars.append(' ')  # Insert a space
        output_chars.append(current_char)

    # Return the recognized sentence with spaces
    recognized_sentence = ''.join(output_chars)
    return recognized_sentence


if __name__ == "__main__":
    
    # Example: run on 'images/sentence1.jpeg'\
    for i in range(1, 11):
        result = extract_characters(f'images/num{i}.jpeg', min_area=1500, pad=30)
        print(f"Recognized number {i}:", result)
    for i in range(1, 3):
        result = extract_characters(f'images/short_sentence{i}.jpeg', min_area=350, pad=25, space_threshold=1.25)
        print(f"Recognized short sentence {i} is:", result)
    for i in range(1, 3):
        result = extract_characters(f'images/medium_sentence{i}.jpeg', min_area=350, pad=25, space_threshold=1.3)
        print(f"Recognized medium sentence {i} is:", result)
    for i in range(1, 3):
        result = extract_characters(f'images/long_sentence{i}.jpeg', min_area=400, pad=25)
        print(f"Recognized long sentence {i} is:", result)


