# Functions to do image transformations
__author__ = "Nina Belyavskaya"

import os.path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def read_gray_image(path):
    # read image from file by path
    image = cv.imread(path)
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def gamma_correction(image, gamma=2.5):
    # gamma >1 brightens, <1 darkens; tune as needed
    image_float = image.astype(np.float32) / 255.0
    image_gamma = np.power(image_float, 1.0 / gamma)
    image_gamma = np.clip(image_gamma * 255, 0, 255).astype(np.uint8)
    return image_gamma


def thresholding(image, blocksize=45, C=5):
    # create binary mask using adaptive thresholding
    mask = cv.adaptiveThreshold(
        image, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        blocksize,  # blockSize, can be tuned
        C    # Lower C to make thresholding less aggressive (darker result)
    )
    return mask


def improve_image(image):
    # adaptive thresholding to handle uneven illumination
    adaptive = cv.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=35,
        C=5
    )
    # Invert so text is black, background is white
    return 255 - adaptive


def integral_sum(intgr, x, y, w, h):
    return int(intgr[y+h, x+w] - intgr[y, x+w] - intgr[y+h, x] + intgr[y, x])


def crop_dark_borders(image):
    mask = thresholding(image)
    (num_labels, labels, stats, _) = cv.connectedComponentsWithStats(mask)
    # label 0 is background
    # sort non-background by area (CC_STAT_AREA)
    sorted_idx = np.argsort(stats[1:, cv.CC_STAT_AREA]) + 1
    idx_largest = sorted_idx[-1]
    stats_largest = stats[idx_largest]
    mask_largest = (labels == idx_largest)

    intgr = cv.integral(mask_largest.astype(np.uint8))
    # how many pixels can be ignored as noise
    noise_threshold = image.shape[1] * 0.05

    bbox = stats_largest[[cv.CC_STAT_LEFT, cv.CC_STAT_TOP,
                          cv.CC_STAT_WIDTH, cv.CC_STAT_HEIGHT]].tolist()
    # could also start with the whole image but this gives it a head start

    while True:
        (x, y, w, h) = bbox
        print(f"bbox: {bbox}")

        if w <= 2 or h <= 2:  # pointless box
            break

        counts = np.array([w, w, h, h])

        in_top = integral_sum(intgr, x,     y,     w, 1)
        in_bottom = integral_sum(intgr, x,     y+h-1, w, 1)
        in_left = integral_sum(intgr, x,     y,     1, h)
        in_right = integral_sum(intgr, x+w-1, y,     1, h)
        ins = np.array([in_top, in_bottom, in_left, in_right])
        outs = counts - ins

        if not (outs > noise_threshold).any():  # nothing to crop anymore (done)
            break

        side_index = np.argmax(outs)

        match side_index:
            case 0: bbox = (x,   y+1, w,   h-1)
            case 1: bbox = (x,   y,   w,   h-1)
            case 2: bbox = (x+1, y,   w-1, h)
            case 3: bbox = (x,   y,   w-1, h)

    return image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


def crop_to_content(image):
    # Find long vertical lines near the sides and crop image between them

    # Use the "improved" image (binarized) for line detection
    mask = thresholding(image)
    # We'll use morphological operations to enhance vertical lines, then find contours

    # Enhance vertical lines
    vertical_kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (1, mask.shape[0] // 10))
    vertical_lines = cv.morphologyEx(
        mask, cv.MORPH_OPEN, vertical_kernel, iterations=1)

    # Find contours of vertical lines
    contours, _ = cv.findContours(
        vertical_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Show detected contours on the vertical_lines image
    contour_image = cv.cvtColor(vertical_lines, cv.COLOR_GRAY2BGR)
    # Filter for long vertical lines near the left and right sides
    height, width = image.shape
    min_line_length = int(0.5 * height)  # at least 50% of image height
    side_margin = int(0.3 * width)      # within 30% from sides

    left_x = None
    right_x = None

    for cnt in contours:
        x, y, cw, ch = cv.boundingRect(cnt)
        if ch > min_line_length:
            # Left side
            if x < side_margin:
                if left_x is None or x + cw > left_x:
                    left_x = x + cw  # crop to the right of the line
            # Right side
            if x + cw > width - side_margin:
                if right_x is None or x < right_x:
                    right_x = x      # crop to the left of the line

    # Fallback if no lines found: use full image
    if left_x is None:
        left_x = 0
    if right_x is None:
        right_x = width

    # Crop between the detected vertical lines
    return image[:, left_x:right_x]


def morphological_operations(image):
    # Use morphological operations to clean up the image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cleaned = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def clean_small_components(image, min_area=30):
    # Remove small connected components (noise)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        image, connectivity=4)
    cleaned = np.zeros_like(image)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def find_contours(image, line_percentil=90, char_percentil=50,
                  proportion_limit=5, noise_size=10):
    # Find contours of characters and filter them based on size percentiles
    kernel = np.ones((5, 5), np.uint8)
    grad = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)

    _, bw = cv.threshold(grad, 0.0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    connected = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(
        connected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes0 = [cv.boundingRect(cnt) for cnt in contours]
    height, width = image.shape
    print("Initial number of detected boxes", len(boxes0))
    boxes = boxes0
    boxes_new = []
    for i, (x, y, w, h) in enumerate(boxes):
        if h < noise_size and w < noise_size:  # Too small box, noise
            continue
        if h > height / 5 and h / w > 3:  # Too high box, vertical line
            continue
        boxes_new.append((x, y, w, h))
    boxes = boxes_new

    character_size = np.percentile([box[3] for box in boxes], char_percentil)
    print(f"Character size ({char_percentil}. percentil): {character_size}")
    big_character_size = np.percentile([box[3] for box in boxes], line_percentil)
    print(f" ({line_percentil}. percentil): {big_character_size}")
    while big_character_size / character_size > proportion_limit:
        boxes_new = []
        for i, (x, y, w, h) in enumerate(boxes):
            if h + w < character_size / 2 + 1:  # Příliš malý box, pravděpodobně šum
                continue
            if h > big_character_size * 3:  # Too big box, split it
                h2 = int(h / 2)
                boxes_new.append((x, y, w, h2+1))
                boxes_new.append((x, y + h2, w, h - h2 + 1))
            else:
                boxes_new.append((x, y, w, h))
        boxes = boxes_new
        # znovu seřadíme podle Y
        character_size = np.percentile([box[3] for box in boxes], char_percentil)
        big_character_size = np.percentile([box[3] for box in boxes], line_percentil)
    return boxes, character_size, big_character_size


def create_mask_image(image, boxes):
    mask_image = np.zeros_like(image)
    for (x, y, w, h) in boxes:
        if w + h > 10:  # ignore very small boxes
            mask_image[y - 1:y + h + 1, x - 1:x + w + 1] = 255
    return mask_image


def combine_with_mask(image, mask):
    # Combine original image with mask to keep only areas in the mask
    combined = cv.bitwise_and(image, image, mask=mask)
    return combined


class Line():
    def __init__(self, image, box, boxes):
        (x, y, w, h) = box
        self.box = (x, y, w, h)
        print(f"Line box: {self.box}, number of boxes: {len(boxes)}")
        self.mask = np.zeros((h, w), dtype=np.uint8)
        for (bx, by, bw, bh) in boxes:
            self.mask[by - y:by - y + bh, bx - x:bx - x + bw] = 255
        self.image = image[y:y+h, x:x+w]
        self.image = np.where(self.mask > 0, self.image, 255)

    def find_contours(self):
        contours, _ = cv.findContours(
            self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        # Adjust contours to the position in the original image
        for cnt in contours:
            for point in cnt:
                point[0][0] += self.box[0]  # x offset
                point[0][1] += self.box[1]  # y offset
        # return contours relative to the whole image
        return contours


def split_lines(image, mask, boxes, character_size, big_character_size):
    # Split the image into lines based on the boxes and gaps between them
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
        
    current_y = 0
    current_box_number = 0
    box = boxes[current_box_number]
    lines = []
    status = "blank"
    counts = 0
    line_width = 0
    line_height = 0
    arr = np.array(mask)
            
    height, width = mask.shape
    # move from top to bottom
    while current_y < height:
        # count white pixels in the line of the mask
        line = arr[current_y, :]
        counts = np.sum(line == 255)  # count white pixels (text)

        line_width = max(line_width, counts)
            
        if status == "blank": 
            if counts > character_size * 6: # line starts
                status = "line"
        elif status == "line":
            line_height += 1
            if (line_height > big_character_size and
                current_y > box[1] + box[3] / 2 and
                (counts < line_width / 2 or 
                line_width - counts > character_size * 4) ):
                print(f"Line ends at y={current_y}, counts={counts}, line_width={line_width}")
                status = "end"
        if status == "end":
            (x, y, w, h) = box
            current_mask = []
            # find all boxes where middle is uper then line's bottom
            while (box[1] + box[3] / 2 < current_y + big_character_size / 4 and
                current_box_number < len(boxes)): # add box to line
                box = boxes[current_box_number]
                current_mask.append(box)
                x = min(box[0], x) 
                y = min(box[1], y)
                x1 = max(box[0] + box[2], x + w)
                y1 = max(box[1] + box[3], y + h)
                w = x1 - x
                h = y1 - y
                current_box_number += 1
                
            
            lines.append(Line(image, (x, y, w, h), current_mask))
            status = "blank"
            line_width = 0
            line_height = 0
        
        current_y += 1
        
    return lines
            
        
def show_image(image, title="Image", cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_image(path, image):
    cv.imwrite(path, image)


def show_lines(image, lines):
    # Create an empty canvas
    canvas = np.copy(image)

    contours = []
    for line in lines:
        (x, y, w, h) = line.box
        # Draw rectangle around each line
        cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 3, 
                     lineType=cv.LINE_AA)
        # Get contours of the line
        subcontours = line.find_contours()
        for contour in subcontours:
            contours.append(contour)
    # Draw contours on the canvas
    cv.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    # big canvas
    plt.figure(figsize=(12, 16))
    plt.imshow(canvas, cmap='gray')
    plt.title("Detected lines")
    plt.axis('off')
    plt.show()
            

def show_separated_lines(lines):
    for i, line in enumerate(lines):
        x, y, w, h = line.box
        print(f"Line {i}: {x}, {y}, {w}, {h}")
        plt.figure(figsize=(10, 2)) 
        plt.imshow(line.image, cmap='gray')
        plt.axis('off')
        plt.title(f"Line {i}")
        plt.show()


def save_lines(image_path, image_filename, start_row_number, lines):
    extension = image_filename.split('.')[-1]
    base_name = image_filename.replace(f'.{extension}', '')
    for i, line in enumerate(lines):        
        output_path = os.path.join(image_path, 
                                   f"{base_name}_line_{i+start_row_number}.{extension}")
        save_image(output_path, line.image)