import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


def DrawChinese(img, text, positive, fontSize=20, fontColor=(0, 255, 0)):
    """
    Draws Chinese text on an image at the specified position.

    Args:
    - img: (numpy.ndarray) The image to draw the text on.
    - text: (str) The Chinese text to draw.
    - positive: (tuple) The (x, y) position for the text.
    - fontSize: (int) The font size for the text (default is 20).
    - fontColor: (tuple) The color of the font (default is green).

    Returns:
    - (numpy.ndarray) The image with the Chinese text drawn on it.
    """
    # Convert the OpenCV image to a PIL image to draw Chinese text
    cv2img = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # OpenCV and PIL have different color order
    pilimg = Image.fromarray(cv2img)

    # Create an ImageDraw object to draw the text on the PIL image
    draw = ImageDraw.Draw(pilimg)

    # Load the font with the specified size (ensure you have a proper Chinese font file)
    font = ImageFont.truetype(
        "utils/MSJHL.TTC", fontSize, encoding="utf-8"
    )  # First argument: font file, second: font size

    # Draw the text on the image at the specified position with the chosen font and color
    draw.text(
        positive, text, fontColor, font=font
    )  # First: position, second: text, third: color, fourth: font

    # Convert the PIL image back to an OpenCV image
    cv2charimg = cv2.cvtColor(
        numpy.array(pilimg), cv2.COLOR_RGB2BGR
    )  # Convert back to OpenCV image

    return cv2charimg
