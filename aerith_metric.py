from T2IMetrics.qga.base import QAMetric
import logging
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

import re

logger = logging.getLogger(__name__)

class AerithMetric(QAMetric):
    def __init__(self, 
                 lm_type: str = "gpt-3.5-turbo",
                 vlm_type: str = "llava",
                 om_det_model: str = "om-det-base",  # OM-DET model name
                 confidence_threshold: float = 0.7,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(lm_type=lm_type, vlm_type=vlm_type, device=device, **kwargs)
        
        # Initialize OM-DET model
        self.confidence_threshold = confidence_threshold
        self.om_det = self._load_om_det_model(om_det_model)
        
        # Initialize CLIP for object matching
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if device:
            self.om_det = self.om_det.to(device)
            self.clip_model = self.clip_model.to(device)

    def _load_om_det_model(self, model_name: str):
        """Load OM-DET model from GitHub repository."""
        try:
            # Clone the repository if not already present
            import os
            if not os.path.exists("om-det"):
                os.system("git clone https://github.com/OM-DET/om-det.git")
            
            # Import the model
            import sys
            sys.path.append("om-det")
            from om_det import OMDET
            
            # Initialize the model
            model = OMDET.from_pretrained(model_name)
            return model
        except Exception as e:
            logger.error(f"Failed to load OM-DET model: {e}")
            raise

    def extract_object_from_question(self, question: str) -> str:
        """Extract the main object from a question."""
        # Simple pattern matching for common question types
        patterns = [
            r"what is the ([\w\s]+)\?",  # "what is the object?"
            r"where is the ([\w\s]+)\?",  # "where is the object?"
            r"how many ([\w\s]+)",       # "how many objects"
            r"is there a ([\w\s]+)",     # "is there an object"
            r"can you see ([\w\s]+)",    # "can you see object"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1).strip()
        
        # Fallback: return the last few words of the question
        return " ".join(question.lower().split()[-3:]).strip()

    def detect_and_crop_object(self, image: Image.Image, object_name: str) -> List[Image.Image]:
        """Detect and crop objects using OM-DET."""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Get object detection from OM-DET
        detections = self.om_det.detect(
            img_array,
            object_name,
            confidence_threshold=self.confidence_threshold
        )
        
        # Get cropped regions
        cropped_images = []
        for detection in detections:
            # Extract bounding box
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop the image
            cropped = image.crop((x1, y1, x2, y2))
            
            # Verify with CLIP
            inputs = self.clip_processor(
                images=cropped,
                text=[f"a photo of {object_name}"],
                return_tensors="pt",
                padding=True
            )
            
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.clip_model(**inputs)
            similarity = outputs.logits_per_image.item()
            
            if similarity > 0.5:
                cropped_images.append(cropped)

        return cropped_images

    def generate_questions(self, prompt: str) -> List[Dict]:
        """Generate questions based on the input prompt.
        
        Args:
            prompt: The input question/prompt
            
        Returns:
            List of dicts containing questions and expected answers
        """
        # Since we're using the input as a question directly,
        # we'll just wrap it in the expected format
        return [{
            'question': prompt,
            'answer': 'yes',  # Default answer, will be processed by VLM
            'choices': ['yes', 'no']  # Binary choices for validation
        }]

    def calculate_score(self, image, prompt: str) -> float:
        """Override base calculate_score to use object detection + VLM pipeline."""
        logger.info(f"Analyzing prompt: {prompt}")
        
        # Generate questions using base class
        questions = self.generate_questions(prompt)
        logger.info(f"Generated {len(questions)} questions")
        
        answers = []
        for q in questions:
            # Extract object from question
            object_name = self.extract_object_from_question(q['question'])
            logger.debug(f"Extracted object: {object_name}")
            
            # Detect and crop objects
            cropped_images = self.detect_and_crop_object(image, object_name)
            
            if not cropped_images:
                # If no objects detected, use original image
                answer = self.vlm.get_answer(q['question'], image)
            else:
                # Use the first detected object
                answer = self.vlm.get_answer(q['question'], cropped_images[0])
            
            answers.append(answer)
            logger.debug(f"Q: {q['question']}, A: {answer}")
        
        # Calculate final score
        return self.compute_score(answers, questions)

    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute score based on VLM answers."""
        if not questions:
            return 0.0

        total_score = 0.0
        valid_questions = 0

        for answer, question in zip(answers, questions):
            is_correct = self.answer_processor.validate_answer(
                answer,
                question['answer'],
                choices=question.get('choices', ['yes', 'no']),
                threshold=self.similarity_threshold
            )
            
            total_score += float(is_correct)
            valid_questions += 1

        return total_score / valid_questions if valid_questions > 0 else 0.0