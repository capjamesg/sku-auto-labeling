from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.core import EmbeddingOntology
from autodistill.core.custom_detection_model import CustomDetectionModel

import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INPUT_FOLDER = "samples"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "album cover"

images = os.listdir("samples")

images_to_classes = {
    "midnights": "IMG_9022.jpeg",
    "men amongst mountains": "323601467684.jpeg",
    "we are": "IMG_9056.jpeg",
    "oh wonder": "Images (5).jpeg",
    "brightside": "Images (4).jpeg",
    "tears for fears": "Images (3).jpeg"
}

embeddings_to_classes = {}

with torch.no_grad():
    for cls, image in images_to_classes.items():
        image = preprocess(Image.open(os.path.join("samples", image))).unsqueeze(0).to(device)
        embeddings_to_classes[tuple(model.encode_image(image).cpu().numpy()[0])] = cls


SAMCLIP = CustomDetectionModel(
    detection_model=GroundingDINO(
        CaptionOntology({PROMPT: PROMPT})
    ),
    classification_model=CLIP(
        EmbeddingOntology(embeddings_to_classes.items())
    )
)

SAMCLIP.label(input_folder=DATASET_INPUT, output_folder=DATASET_OUTPUT, extension=".jpeg")