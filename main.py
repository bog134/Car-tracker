from car_recognition import CarRecognizer
from car_tracking import CarTracker
from config import CONFIG

def main():
    # Initialize car recognizer
    car_recognizer = CarRecognizer(CONFIG['car_features'])
  
    # Recognize cars in the image
    car_recognizer.recognize_cars_in_image(CONFIG['image_path'])
                                                   
    # Initialize car tracker
    object_tracker = CarTracker(CONFIG['car_features'])
                                                   
    # Track cars in the videos
    for i,video_path in enumerate(CONFIG['video_paths']):                                                                            
        object_tracker.track_cars_in_video(video_path,f'Car tracking {i+1}')
          
if __name__ == "__main__":
    main()                       