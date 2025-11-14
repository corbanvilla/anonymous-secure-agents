from enum import Enum


class IntegrityLabel(str, Enum):
    DEVELOPER = "developer"
    USER = "user"
    THIRD_PARTY = "third_party"


class CapabilityLabel(str, Enum):
    VIEWABLE = "viewable"
    INTERACTABLE = "interactable"
