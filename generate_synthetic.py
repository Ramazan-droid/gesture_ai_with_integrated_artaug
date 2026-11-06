
import os
import random
import itertools
import torch

from diffsynth import (
    ModelManager,
    FluxImagePipeline,
    download_customized_models
)


# Configuration

OUT_DIR = r"C:\Users\ramaz\Desktop\compvision\asik1\data\synthetic"

# Start small to test whether CPU inference works.
# Change to 200 after a successful test.
IMAGES_PER_CLASS = 5

DEVICE = "cpu"
DTYPE = torch.float32

random.seed(42)
torch.manual_seed(42)

# Limit CPU threads to avoid excessive thread overhead.
torch.set_num_threads(4)


# Hand gesture prompts

CLASS_PROMPTS = {

    "fist": (
        "a realistic photograph of one human hand "
        "making a closed fist gesture, all fingers curled "
        "into the palm, thumb resting outside the fingers"
    ),

    "peace": (
        "a realistic photograph of one human hand "
        "making a peace sign gesture, index and middle "
        "fingers raised in a V shape, other fingers folded"
    ),

    "stop": (
        "a realistic photograph of one human hand "
        "making a stop gesture, open palm facing directly "
        "toward the camera, all five fingers extended and separated"
    )

}


# Visual conditions

LIGHTING = [
    "soft natural daylight",
    "harsh midday sunlight",
    "golden hour lighting",
    "overcast lighting",
    "dim indoor lighting",
    "studio lighting",
    "backlit scene"
]

COMPOSITION = [
    "close-up photograph of the hand",
    "hand positioned slightly off-center",
    "hand photographed from a slightly lower angle",
    "hand photographed from a slightly higher angle",
    "hand centered in the frame",
    "hand occupying most of the image"
]

BACKGROUND = [
    "against a plain white wall",
    "in a living room",
    "against a dark background",
    "in an outdoor environment",
    "against a blurred office background",
    "in a classroom"
]


# Load ArtAug LoRA

print("Downloading ArtAug LoRA...")

lora_path = download_customized_models(
    model_id="DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1",
    origin_file_path="merged_lora.safetensors",
    local_dir="models/lora" 
)[0]


# Initialize FLUX on CPU

print("Loading FLUX.1-dev on CPU...")
print("This may require substantial RAM and time.")

model_manager = ModelManager(
    torch_dtype=DTYPE,
    device=DEVICE,
    model_id_list=["FLUX.1-dev"]
)

model_manager.load_lora(
    lora_path,
    lora_alpha=1.0
)

pipe = FluxImagePipeline.from_model_manager(
    model_manager
)

print("Model loaded!")


# Generate synthetic images

os.makedirs(OUT_DIR, exist_ok=True)

combos = list(
    itertools.product(
        LIGHTING,
        COMPOSITION,
        BACKGROUND
    )
)

for class_name, base_prompt in CLASS_PROMPTS.items():

    print(f"\nGenerating class: {class_name}")

    class_dir = os.path.join(
        OUT_DIR,
        class_name
    )

    os.makedirs(class_dir, exist_ok=True)

    selected_combinations = random.sample(
        combos,
        IMAGES_PER_CLASS
    )

    for i, (light, composition, background) in enumerate(
        selected_combinations
    ):

        prompt = (
            f"{base_prompt}, "
            f"{composition}, "
            f"{background}, "
            f"{light}, "
            "realistic photography, "
            "natural human hand anatomy, "
            "single hand, sharp focus"
        )

        print(
            f"[{class_name}] "
            f"Generating image {i + 1}/{IMAGES_PER_CLASS}"
        )

        image = pipe(
            prompt=prompt,
            seed=42 + i,
            height=512,
            width=512
        )

        image_path = os.path.join(
            class_dir,
            f"{class_name}_{i:04d}.jpg"
        )

        image.save(image_path)

        print("Saved:", image_path)


print("\nGeneration completed!")
print("Output directory:", OUT_DIR)
