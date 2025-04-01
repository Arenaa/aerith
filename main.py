import sys
from pathlib import Path
from PIL import Image
import torch
from aerith_metric import AerithMetric

def main():
    # Initialize the metric with supported VLM type
    metric = AerithMetric(
        lm_type="openai",
        vlm_type="instructblip",  # Changed from llava to instructblip
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load an image
    # Download a sample image
    import requests
    from io import BytesIO
    
    url = "https://huggingface.co/datasets/saxon/T2IScoreScore?image-viewer=F2BC37D93C30973A907FFB1EDC36DB6A4C33F8A7"
    response = requests.get(url)
    image_path = BytesIO(response.content)
    image = Image.open(image_path)
    
    # Your question
    question = "Does the Christmas tree have lights?	"
    
    # Get scores
    scores = metric(image, question)
    
    # Print results
    print("\nAerith Scores:")
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score:.3f}")

if __name__ == "__main__":
    main() 