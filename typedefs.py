from typing import TypedDict


class Metadata(TypedDict):
    image_id: str
    caption: str
    tags: list[str]
    location: str


class Element(TypedDict):
    id: str
    embedding: list[float]
    metadata: Metadata