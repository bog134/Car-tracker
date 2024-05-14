import cv2
import numpy as np
from utils import find_color_coefficient, Line

class CarTracker:
    def __init__(self, car_features):
        """Initializes car features."""
        self.car_features = car_features

    def track_cars_in_video(self, video_path,video_name):
        """Tracks cars in a video."""

        try:
            video = cv2.VideoCapture(video_path)
            if video is None:
                raise FileNotFoundError(f"Video not found at: {video_path}")
            else:        
                ret, previous_frame = video.read()
                self._detect_finish_line(previous_frame)
                self.crossed_cars = []
                while ret:
                    ret, frame = video.read()
                    if not ret:
                        break
                    # Find car contours
                    car_contours = self._find_car_contours(frame, previous_frame)
                    # Track trajectories and detect finish line crossing
                    frame1 = self._track_cars(frame, car_contours)

                    # Display the video frame
                    cv2.imshow(video_name, frame1)
                    if cv2.waitKey(10) == ord('q'):
                        break

                    previous_frame = frame.copy()
                video.release()
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"An error occurred during car tracking: {e}")

    def _detect_finish_line(self,frame):
        finish = frame.copy()

        finish_canny = cv2.Canny(finish,160,200)
        
        finish_dilate = cv2.dilate(finish_canny, np.ones((5,5), np.uint8))

        finish_conts,_=cv2.findContours(finish_dilate,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        good_conts =[]

        for cont in finish_conts:
            area= cv2.contourArea(cont)
            if area>9000:
                [vx,vy,x,y] = cv2.fitLine(cont, cv2.DIST_L2,0,0.01,0.01)
                self.finish_line = Line(vx,vy,x,y)
                good_conts.append(cont)

    def _find_car_contours(self, frame, previous_frame):
        """Finds car contours in the video frame."""

        diff = cv2.absdiff(frame, previous_frame)  # Absolute difference between frames
        blured = cv2.GaussianBlur(diff, (7,7), 0)
        gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
        dilated = cv2.dilate(thresh, np.ones((5,5), np.uint8))  # Adjust dilation kernel as needed
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
       
        car_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15000 > area > 4000:  # Adjust minimum area threshold as needed
                car_contours.append(contour)

        return car_contours 
    
    def _track_cars(self, frame, car_contours):
        """Tracks car trajectories and detects finish line crossing."""
        frame1 = frame.copy()
        for contour in car_contours:
            # Check if contour crosses finish line
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if x <= self.finish_line.x[0] and x + 0.1 * w >= self.finish_line.x[0]:
                # Draw finish line
                _, cols = frame.shape[:2]
                lefty = int((-self.finish_line.x * self.finish_line.vy / self.finish_line.vx) + self.finish_line.y)
                righty = int(((cols - self.finish_line.x) * self.finish_line.vy / self.finish_line.vx) + self.finish_line.y)
                cv2.line(frame1, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

                # Identify car and print name if not already printed
                car_name = self._identify_car(frame, contour)
                if car_name not in self.crossed_cars:
                    self.crossed_cars.append(car_name) 

        for i,car_name in enumerate(self.crossed_cars):
            cv2.putText(frame1, f'{i+1}. {car_name}', (10, (i+1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        
        return frame1
    
    def _identify_car(self, frame, contour):
        """Identifies the car type based on its features."""
        x, y, _, h = cv2.boundingRect(contour)
        
        for i in range(len(self.car_features)):        
            w1 = int(self.car_features[i]['width_height_pro'] * h)

            color_intensity = find_color_coefficient(frame, x, y, w1, h)
            similarity = self._find_bolid_contours(frame, x, y, w1, h)
        
            # Compare color_intensity with predefined ranges in car_features
            if color_intensity[1] < self.car_features[1]['color_intensity'][1] + 10 and \
            color_intensity[1] > self.car_features[1]['color_intensity'][1] - 10:
                if similarity < 0.08:
                    return 'bolid'
                else:
                    return 'ferrari'
            elif color_intensity[2] < self.car_features[2]['color_intensity'][2] + 10 and \
                color_intensity[2] > self.car_features[2]['color_intensity'][2] - 10:
                return 'bmw'
                
        return 'unknown'
        
    def _find_bolid_contours(self,frame, x, y, w, h):
        car_img = frame[y:y+h, x:x+w].copy()
        gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 65 , 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(thresh, np.ones((3,3), np.uint8))
        contour, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        similarity = cv2.matchShapes(contour[0], self.car_features[0]['contours'][0], 2, 0.0)
        return similarity