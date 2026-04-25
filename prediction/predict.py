import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import pickle
import random

class DogCat:
    def __init__(self, filename):
        self.filename = filename
        self.class_labels = [
            'african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox',
            'armadillo', 'baboon', 'badger', 'blue_whale', 'brown_bear', 'camel', 'cat',
            'dog', 'dolphin', 'giraffe', 'groundhog', 'highland_cattle', 'horse', 'jackal', 
            'kangaroo', 'koala', 'manatee', 'mongoose', 'mountain_goat', 'opossum', 'orangutan',
            'otter', 'polar_bear', 'porcupine', 'red_panda', 'rhinoceros', 'sea_lion', 'seal',
            'snow_leopard', 'squirrel', 'sugar_glider', 'tapir', 'vampire_bat', 'vicuna', 
            'walrus', 'warthog', 'water_buffalo', 'weasel', 'wildebeest', 'wombat', 'yak', 'zebra'
        ]
        # Load the model and database
        self.model = load_model(
            os.path.join("model", "best_model_weights.keras"), 
            compile=False,
            safe_mode=False
        )
        database_path = os.path.join("model", "image_database.pkl")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)

    def predictiondogcat(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(test_image)
        predicted_index = np.argmax(prediction)
        predicted_label = self.class_labels[predicted_index]
        
        return predicted_label  # Return string instead of list
    
    def get_similar_images(self, num_images=5):
        # Predict the class of the input image
        predicted_label = self.predictiondogcat()
        
        # Retrieve similar images from database
        if predicted_label in self.database:
            class_images = [img_path for features, img_path in self.database[predicted_label]]
            if class_images:
                retrieved_images = random.sample(class_images, min(len(class_images), num_images))
                return predicted_label, retrieved_images
        return predicted_label, []  # Return string instead of list