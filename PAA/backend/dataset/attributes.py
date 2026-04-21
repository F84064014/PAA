import numpy as np
from collections import defaultdict

class Attributes:
    def __init__(self, attributes: list[str]) -> None:
        self.attributes = attributes
        self.grouped_attributes = \
            self.build_grouped_attributes()

    def build_grouped_attributes(self):
        grouped_attributes = defaultdict(list)
        for idx, attribute in enumerate(self.attributes):
            group, attr = attribute.rsplit('-', 1)
            grouped_attributes[group].append((idx, attr))
        return grouped_attributes

    def list(self) -> list[str]:
        return self.attributes
    
    def group(self) -> dict:
        """
        return a dict[GroupName] = list[(Attr_i, AttrName)]
        """
        return self.grouped_attributes.items()
    
    def __len__(self):
        return len(self.attributes)