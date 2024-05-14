CONFIG = {
    'image_path': './data/P1380295.jpg',  # Path to the image used for car recognition
    'video_paths': ['./data/P1380292.MOV', './data/P1380293.MOV'],  # List of paths to videos for tracking
    'car_features': [
        {'name': 'bolid', 'contours': None, 'width_height_pro': 0},
        {'name': 'ferrari', 'color_intensity': None, 'width_height_pro': 0},
        {'name': 'bmw', 'color_intensity': None, 'width_height_pro': 0},
    ]  # List of car features for recognition
}