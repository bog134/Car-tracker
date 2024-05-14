import cv2
import numpy as np
import imutils
from utils import find_color_coefficient

class CarRecognizer:
    def __init__(self, car_features):
        """Initializes car features."""
        self.car_features = car_features

    def recognize_cars_in_image(self, image_path):
        """Recognizes cars in an image."""

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at: {image_path}")
            else:
                # Rotate the image
                rotated_image = self._rotate_image(image)
                # Resize the image
                resized_image = cv2.pyrDown(rotated_image)
                # Find car contours
                car_contours, bounding_boxes = self._find_car_contours(resized_image)
                # Extract car features
                self._extract_car_features(resized_image, car_contours, bounding_boxes)
                # Display the image with marked cars
                self._display_results(resized_image, car_contours, bounding_boxes)

        except Exception as e:
            print(f"An error occurred during car recognition: {e}")

    def _rotate_image(self, image, angle=15):
        """Rotates the image by the given angle."""
        return imutils.rotate(image, angle)

    def _find_car_contours(self, image):
        """Finds car contours in the image."""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(thresh, np.ones((17, 17), np.uint8))
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        car_contours = []
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 40000 < area < 50000:
                car_contours.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
        return car_contours, bounding_boxes

    def _extract_car_features(self, image, car_contours, bounding_boxes):
        """Extracts car features."""

        for i, _ in enumerate(car_contours):
            x, y, w, h = bounding_boxes[i]
            car_name = self.car_features[i]['name']
            self.car_features[i]['width_height_pro'] = w/h
            if car_name == 'bolid':
                self.car_features[i]['contours'] = self._find_bolid_contours(image, x, y, w, h)
            else:
                color_intensity = find_color_coefficient(image, x, y, w, h)
                self.car_features[i]['color_intensity'] = color_intensity

    def _find_bolid_contours(self,image, x, y, w, h):
        """Finds characteristic contours for the 'bolid' car."""

        bolid_image = image[y:y+h, x:x+w].copy()
        gray = cv2.cvtColor(bolid_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _display_results(self, image, car_contours, bounding_boxes):
        """Displays the image with marked cars."""

        for i, _ in enumerate(car_contours):
            x, y, w, h = bounding_boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            car_name = self.car_features[i]['name']
            cv2.putText(image, car_name, (x + w, int(y + h * 0.75)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
        cv2.imshow("Car Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()