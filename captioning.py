# This script uses qwen2.5vl:3b model to generate captions for images
# It uses ollama library to interact with the model
# Refer to the installation instructions in the README.md file to set up the models
import json
import re
import ollama
import os
import time


class ImageCaptioner:
    # Make the model selection configurable for different models
    # NOTE: Make sure that the models is available in the ollama library and supports the image captioning task
    def __init__(self, model_name="qwen2.5vl:3b"):
        self.model_name = model_name
        self.system_prompt = """
        You are an expert image captioning model focused on urban safety, sanitation, and infrastructure.
        Describe this image by providing:
        1. A detailed caption describing what is visible and relevant to urban conditions.
        2. A list of relevant tags highlighting key features, objects, or issues (e.g., "flooding", "trash", "pedestrian lane", "damaged road", "congested lanes", "risk of flooding", "accident prone").

        Return only valid JSON in this format:
        {
        "caption": "<caption here>",
        "tags": ["tag1", "tag2", ...]
        }

        Do not include extra explanations or text. Focus on relevant visual cues.
        """

        # Initialize the model
        self.init_model()

    def init_model(self):
        # Here we want to set the model to its "warm start" state
        # We do not want to make the first request to the model to suffer from cold start
        # We prompt the model with a simple question to warm it up
        ollama.generate(
            model=self.model_name,
            prompt="hi",
            stream=False,
            options={
                "temperature": 0.3,
                "num_predict": 500
            }
        )

    def generate_caption(self, image_path: str) -> str | None:
        try:
            # Ensure path is valid
            if not os.path.exists(image_path):
                exception_message = f"Image path '{image_path}' does not exist."
                raise FileNotFoundError(exception_message)

            # Use Ollama SDK to generate the caption
            result = ollama.generate(
                model=self.model_name,
                prompt=self.system_prompt,
                images=[image_path],
                stream=False,
                options={
                    "temperature": 0.3,
                    "num_predict": 500
                }
            )
            # Return only the text caption from the response
            return result["response"]
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None


def load_manifest(manifest_path: str) -> list | None:
    with open(manifest_path, "r") as f:
        manifest = f.read()
        manifest = json.loads(manifest)
        return manifest
    return None


def verify_manifest(manifest_path: str):
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print("Manifest file not found.")
        return

    for image in manifest:
        image_path = image["image"]
        if not os.path.exists(image_path):
            print(f"Image '{image_path}' not found.")
            return


def extract_json_from_response(response_text):
    """
    Extracts the first JSON object from a string.
    """
    try:
        # Use regex to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response.")
    except json.JSONDecodeError as e:
        raise ValueError("JSON decoding failed.") from e


def generate_captions(manifest_path: str):
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print("Manifest file not found.")
        return

    captioner = ImageCaptioner(model_name="gemma3:4b")
    caption_manifest = []

    print("Generating caption manifest...")
    for image in manifest:
        image_path = image["image"]
        caption = captioner.generate_caption(image_path)
        if caption is None:
            print(f"Failed to generate caption for image '{image_path}'.")
            continue

        # Parse the caption and tags from the response
        caption_json = extract_json_from_response(caption)
        caption = caption_json["caption"]
        tags = caption_json["tags"]

        # Add the caption and tags to the manifest
        image["caption"] = caption
        image["tags"] = tags
        caption_manifest.append(image)
        print("Caption generated for image:", image_path)

    with open("caption_manifest.json", "w") as f:
        json.dump(caption_manifest, f, indent=2)
    print("Caption manifest generated.")


def main():
    manifest_path = "data/image_manifest.json"
    # run the manifest verification
    verify_manifest(manifest_path)
    # Run the captioning
    generate_captions(manifest_path)


if __name__ == "__main__":
    main()
