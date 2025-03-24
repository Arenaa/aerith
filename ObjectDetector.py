import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline
import yolov5
import cv2
import numpy as np

class ObjectDetectionQA:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.object_detector = yolov5.load('yolov5s.pt')  # Using small model variant, will download automatically
        
        # Initialize VQA pipeline
        self.vqa_pipeline = pipeline("visual-question-answering", 
                                   model="dandelin/vilt-b32-finetuned-vqa")

    def load_image(self, image_path):
        """Load image from path or URL"""
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        return image

    def detect_objects(self, image, prompt):
        """Detect objects in the image using YOLO"""
        # Run detection
        results = self.object_detector(image)
        
        # Process detection results
        detected_objects = []
        for pred in results.pred[0]:
            confidence = pred[4].item()
            class_id = int(pred[5].item())
            label = results.names[class_id]
            
            if confidence > 0.5:  # confidence threshold
                detected_objects.append(label)
        
        return list(set(detected_objects))  # Remove duplicates

    def answer_question(self, image, prompt, detected_objects):
        """Answer questions about detected objects using VQA"""
        # Filter prompt based on detected objects
        relevant = any(obj.lower() in prompt.lower() for obj in detected_objects)
        
        if not relevant:
            return "No relevant objects detected for this question."
        
        # Get answer from VQA model
        answer = self.vqa_pipeline(image, prompt)
        return answer['answer']

def main():
    # Initialize the class
    od_qa = ObjectDetectionQA()
    
    # Get user inputs
    image_path = input("Enter image path or URL: ")
    prompt = input("Enter your question about the image: ")
    
    try:
        # Load and process image
        image = od_qa.load_image(image_path)
        
        # Detect objects
        detected_objects = od_qa.detect_objects(image, prompt)
        print("\nDetected objects:", ", ".join(detected_objects))
        
        # Get answer
        answer = od_qa.answer_question(image, prompt, detected_objects)
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
