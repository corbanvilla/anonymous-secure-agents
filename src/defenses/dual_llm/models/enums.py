from enum import Enum


class ElementContext(str, Enum):
    """Contexts describing where an element lives on the page."""

    HEAD = "head"
    HEADER = "header"
    NAVIGATION = "navigation"
    MAIN = "main"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    FORM = "form"
    ADVERTISEMENT = "advertisement"
    MODAL = "modal"
    IFRAME = "iframe"
    USER_CONTENT = "user_content"
    THIRD_PARTY_WIDGET = "third_party_widget"
    OTHER = "other"


class ElementPurpose(str, Enum):
    """Purposes describing the role of an element."""

    CONTENT = "content"
    LINK = "link"
    STYLE = "style"
    SCRIPT = "script"
    MEDIA = "media"
    FORM_CONTROL = "form_control"
    STRUCTURE = "structure"
    AD = "advertising"
    TRACKING = "tracking"
    AUTHENTICATION = "authentication"
    COMMENT = "comment"
    EMBED = "embed"
    OTHER = "other"


class IntegrityLabel(str, Enum):
    DEVELOPER = "developer"
    USER = "user"
    THIRD_PARTY = "third_party"


class RelevanceLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
