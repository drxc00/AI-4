from sentence_transformers import SentenceTransformer
from store import VectorStore
import json
import ollama
from typedefs import Element, Metadata


class Rag:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2')
        self.store = VectorStore()
        self.system_prompt = """
        You are a helpful assistant trained to analyze urban environments through images and textual descriptions. 
        Your primary goal is to support Sustainable Development Goals (SDGs) by identifying and reasoning about issues related to:
        - Urban safety and pedestrian infrastructure
        - Road and sidewalk conditions
        - Traffic congestion and accessibility
        - Environmental cleanliness (e.g. trash, debris, flooding)
        - Informal urban development or street vendors

        You will be given a user question, along with a set of image-derived data including captions, tags, and location metadata. 
        Your task is to synthesize this information and answer the user's question clearly and informatively.

        ONLY use the provided evidence. Do not assume or invent information beyond what is shown in the image descriptions and tags.

        Respond in a helpful, analytical, and concise manner.
        """

    def embed(self, t: str) -> list[float]:
        return self.embedding_model.encode(t).tolist()

    def build_text(self, meta: Metadata) -> str:
        # Format the metadata elemen into a string for the text embedding
        sf = """
        Caption: {caption}
        Tags: {tags}
        Location: {location}
        """
        text = sf.format(**meta)
        return text

    def load_manifest(self, manifest_path: str):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            for element in manifest:
                image_id = element["image_id"]
                image_path = element["image"]
                caption = element["caption"]
                tags = element["tags"]
                location = element["location"]
                metadata = Metadata(
                    image_id=image_id,
                    image_path=image_path,
                    caption=caption,
                    tags=tags,
                    location=location
                )

                t = self.build_text(metadata)
                embedding = self.embed(t)
                element = Element(
                    id=image_id,
                    embedding=embedding,
                    metadata=metadata
                )
                self.store.add(element)

    def create_prompt(self, q: str, elements: list[Element]) -> str:
        prompt = f"""
        You are an urban analysis assistant helping to answer questions based on images related to Sustainable Development Goals (SDGs), 
        such as urban infrastructure, safety, sanitation, and traffic.
        The user asked: "{q}"
        Below are the most relevant images retrieved, each with its caption, tags, and location. Use these to answer the question accurately, citing key visual indicators.
        """
        for i, e in enumerate(elements, 1):
            meta = e["metadata"]
            prompt += (
                f"Image {i}:\n"
                f"Location: {meta['location']}\n"
                f"Caption: {meta['caption']}\n"
                f"Tags: {', '.join(meta['tags'])}\n\n"
            )
        prompt += (
            "Based on the above information, provide a concise and informative answer to the user's question."
        )
        return prompt

    def generate(self, prompt: str) -> str:
        # Generate a response from the llm
        response = ollama.generate(
            model="gemma3:1b",
            system=self.system_prompt,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.3,
                "num_predict": 500
            }
        )

        return response["response"]

    def ask(self, q: str, k: int):
        if not self.store.store:
            print(
                "No elements in store. Make sure to load the caption manifest first using load_manifest method.")
            return

        # Get query embedding
        query_embedding = self.embed(q)

        # Query the store for top-k most similar elements
        results = self.store.query(query_embedding, k)

        # Create a prompt for the LLM
        prompt = self.create_prompt(q, results)

        # Generate a response from the LLM
        response = self.generate(prompt)

        return {
            "response": response,
            "sources": [e["metadata"] for e in results]
        }
