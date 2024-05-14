class Line:
    """Represents a straight line."""
    def __init__(self, vx, vy, x, y):
        self.vx = vx  # Direction vector x-component
        self.vy = vy  # Direction vector y-component
        self.x = x    # Point on the line, x-coordinate
        self.y = y    # Point on the line, y-coordinate

def find_color_coefficient(image, x, y, w, h):
    """Calculates the color intensity coefficient in a region of the image."""

    color_intensity_r = 0
    color_intensity_g = 0
    color_intensity_b = 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            color_intensity_r += int(image[j][i][2])  
            color_intensity_g += int(image[j][i][1])
            color_intensity_b += int(image[j][i][0])
    return (color_intensity_r / (w * h),color_intensity_g / (w * h),color_intensity_b / (w * h))