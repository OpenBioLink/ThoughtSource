from dataclasses import dataclass

import datasets

@dataclass
class ThoughtSourceConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None
