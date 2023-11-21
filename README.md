# SKU Auto Labeling (Computer Vision)

<img width="843" alt="Screenshot 2023-11-20 at 19 00 00" src="https://github.com/capjamesg/sku-auto-labeling/assets/37276661/6d4fd1f3-80ea-4d37-8229-6eafe0935b26">

Above, inference results from a trained model that uses data labeled with this project.

Use Grounding DINO and CLIP to auto-label SKUs and product covers.

## Project Description

This project uses [Autodistill](https://github.com/autodistill/autodistill) to connect two models for auto-labeling: Grounding DINO and CLIP. Grounding DINO detects objects in images, and CLIP refines the predictions to a more specific class.

This project lets you provide images as a reference for classification, instead of using text prompts. This is ideal if you need to auto-label product SKUs or covers, where you have a reference image for each class.

For example, you can use Grounding DINO to detect vinyl record covers, then use CLIP to classify the record covers into specific album names. This could be used to auto-label a whole dataset of vinyl records.

## Getting Started

First, clone this repository and install the required dependencies:

```bash
git clone https://github.com/capjamesg/sku-auto-labeling
cd sku-auto-labeling
pip install -r requirements.txt
```

The `app.py` script is used for auto-labeling using Grounding DINO and CLIP.

Update the `images_to_classes` dictionary to map class names to reference image names. Set `INPUT_FOLDER` to the name of the folder in which your reference images are stored.

Here is an example class mapping for record cover labeling:

```python
images_to_classes = {
    "midnights": "IMG_9022.jpeg",
    "men amongst mountains": "323601467684.jpeg",
    "we are": "IMG_9056.jpeg",
    "oh wonder": "Images (5).jpeg",
    "brightside": "Images (4).jpeg",
    "tears for fears": "Images (3).jpeg"
}
```

Next, set:

1. `DATASET_INPUT`: The name of the folder containing the images to auto-label.
2. `DATASET_OUTPUT`: The name of the folder to store the auto-labeled images.
3. `PROMPT`: The prompt to use for Grounding DINO. This should be a short phrase that describes the class you want to auto-label.

Finally, run the script:

```bash
python3 app.py
```

The script will label all images in the `DATASET_INPUT` folder and save them to the `DATASET_OUTPUT` folder.

There may be corrections required before you can train a model with your dataset. To review annotations, I recommend uploading them to [Roboflow](https://roboflow.com) and manually auditing data.

## Example Project

I trained a [vinyl record detection model](https://app.roboflow.com/capjamesg/records-autodistill/deploy/2) with this script. With 42 images used for training, and around two minutes of manual data correction, the model achieved a 95.3% mAP.

## License

This project is licensed under an [MIT license](LICENSE).
