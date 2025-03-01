import torch
import timm
import tensorflow as tf
from torch import nn
from torchvision import transforms
import numpy as np
from PIL import Image
from deepface import DeepFace
import cv2
from torch.cuda.amp import autocast
import joblib
import os
from mtcnn import MTCNN
from EmotionClassifier import EmotionClassifier

from deepface.models.facial_recognition import Facenet
from tensorflow.keras import layers, models

class EmotionDetectionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transformations
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        self.facial_model = None
        self.general_model = None
        self.emotion_class_facial = ['angry', 'fear', 'happy', 'sad']
        self.emotion_class_general = ['angry', 'awe', 'fear', 'happy', 'sad']
        
    def load_models(self, facial_model_path='Model/FacialEmotionModel.h5',
                    general_model_path='Model/NonFacialEmotionModelV2.pth'):
        """Load both facial and general emotion models"""
        # Load facial emotion model (TensorFlow/Keras)
        #self.facial_model = tf.keras.models.load_model(facial_model_path)
        
        # Load general emotion model (PyTorch)
        self.general_model = EmotionClassifier(num_classes=5)
        self.general_model.to(self.device)
        checkpoint = torch.load(general_model_path, map_location=self.device)
        self.general_model.load_state_dict(checkpoint['model_state_dict'])
        self.general_model.eval()
        
    def save_system(self, save_path='Model/Emotion_Detection_System_cpu.joblib'):
        """Save the entire system using joblib"""
        # Ensure general_model state_dict is on CPU
        state_dict = self.general_model.state_dict()
        state_dict = {k: v.to('cpu') for k, v in state_dict.items()}
        system_state = {
            'facial_model_weights': self.facial_model.get_weights(),
            'general_model_state': state_dict,
            'emotion_class_facial': self.emotion_class_facial,
            'emotion_class_general': self.emotion_class_general,
            'transform_state': self.val_transform.state_dict() if hasattr(self.val_transform, 'state_dict') else None
        }
        joblib.dump(system_state, save_path)
        print(f"System saved to {save_path}")
        
    def load_system(self, load_path='Model/Emotion_Detection_System_cpu.joblib'):
        """Load the entire system from joblib"""
        system_state = joblib.load(load_path)
        
        # Recreate and load facial model
        # self.facial_model = tf.keras.models.load_model('Model/FacialEmotionModel.h5')
        self.facial_model = create_emotion_model(num_classes=4, dropout_rate=0.2)  # Match your training setup
        self.facial_model.set_weights(system_state['facial_model_weights'])
        
        # Recreate and load general model
        self.general_model = EmotionClassifier(num_classes=5)
        self.general_model.to(self.device)  # Device is set to 'cpu' in __init__
        
        # Load the state dictionary, mapping to CPU explicitly
        state_dict = system_state['general_model_state']
        # Ensure all tensors are moved to CPU
        state_dict = {k: v.to('cpu') for k, v in state_dict.items()}
        self.general_model.load_state_dict(state_dict)
        self.general_model.eval()
        
        # Load other attributes
        self.emotion_class_facial = system_state['emotion_class_facial']
        self.emotion_class_general = system_state['emotion_class_general']
        
        print("System loaded successfully")

    def create_emotion_model(num_classes=4, dropout_rate=0.2):
        """Recreate the facial emotion model architecture"""
        base_model = Facenet.load_facenet512d_model()  # Assuming Facenet is available
        
        # Add more sophisticated top layers
        x = base_model.layers[-2].output
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(dropout_rate)(x)
        # x = layers.Dense(256, activation='relu')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=x)
    
        # Progressive unfreezing strategy (match training setup)
        for layer in model.layers[:-6]:  # Keep more layers frozen initially
            layer.trainable = False
            
        return model

    def detect_face_emotion(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB as done previously
        img = cv2.resize(img, (160, 160)) / 255.0
        predictions = self.facial_model.predict(np.expand_dims(img, axis=0))
        emotion = np.argmax(predictions)
        return self.emotion_class_facial[emotion]

    def detect_general_emotion(self, image_path):
        """Detect emotion using general model"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with autocast():
                outputs = self.general_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        
        return self.emotion_class_general[predicted.item()], confidence.item()

    def is_face(self, image_path):
        """Check if the image contains a face using MTCNN"""
        try:
            detector = MTCNN()
            image = Image.open(image_path).convert('RGB')
            faces = detector.detect_faces(np.array(image))
            return len(faces) > 0
        except Exception as e:
            print(f"Error in Face Detection: {str(e)}")
            return False
        
    def is_faceV2(self,image_path):
        try:
            # 使用 DeepFace 检测人脸
            result = DeepFace.extract_faces(image_path, detector_backend='ssd')
            return len(result) > 0
        except:
            return False

    def predict_emotion(self, image_path):
        """Main prediction method combining both approaches"""
        has_face = self.is_faceV2(image_path)
        
        if has_face:
            emotion = self.detect_face_emotion(image_path)
            return {
                'type': 'facial',
                'emotion': emotion,
                'confidence': None  # Facial model doesn't provide confidence
            }
        else:
            emotion, confidence = self.detect_general_emotion(image_path)
            return {
                'type': 'general',
                'emotion': emotion,
                'confidence': float(confidence)
            }
