# This script uses qwen2.5vl:3b model to generate captions for images
# It uses ollama library to interact with the model
# Refer to the installation instructions in the README.md file to set up the models
import json
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


def verify_manifest(manifest_path: str):
    with open(manifest_path, "r") as f:
        manifest = f.read()
        manifest = json.loads(manifest)
        print(f"Manifest loaded: {len(manifest)} images.")
        for image in manifest:
            image_path = image["image"]
            if not os.path.exists(image_path):
                print(f"Image '{image_path}' does not exist.")
                return False
    print("Manifest verified.")
    return True


def main():
    # run the manifest verification
    manifest_path = "data/image_manifest.json"
    verify_manifest(manifest_path)
    
    # print("Initializing model...")
    # captioner = ImageCaptioner(model_name="gemma3:4b")
    


if __name__ == "__main__":
    main()
