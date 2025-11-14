from typing import List, Literal, Optional, Type

from pydantic import BaseModel, Field, computed_field, create_model

from .enums import ElementContext, ElementPurpose, IntegrityLabel, RelevanceLabel


class Decision(BaseModel):
    bid: int = Field(..., description="Unique block identifier")
    rationale: str = Field(
        ...,
        description="Rationale for allowing the bid",
    )


class AllowedBids(BaseModel):
    bids: List[Decision] = Field(
        default_factory=list,
        description="List of allowed bids",
    )

    @computed_field
    def allowed_bids(self) -> List[int]:
        return [decision.bid for decision in self.bids]


class ElementMetadata(BaseModel):
    bids: List[int] = Field(..., description="Numeric block identifier")
    context: ElementContext = Field(..., description="Where this element lives or applies")
    purpose: ElementPurpose = Field(..., description="The intention or role of this element")


class RelevanceLabels(BaseModel):
    high: List[int] = Field(
        default_factory=list,
        description="List of block IDs with high relevance",
    )
    medium: List[int] = Field(
        default_factory=list,
        description="List of block IDs with medium relevance",
    )
    low: List[int] = Field(
        default_factory=list,
        description="List of block IDs with low relevance",
    )


def create_explicit_bid_model(
    max_bid_number: int, default_level: RelevanceLabel = RelevanceLabel.HIGH
) -> Type[BaseModel]:
    """Creates a model with explicit bid_0, bid_1, etc. fields"""
    field_definitions = {}

    for bid_id in range(max_bid_number + 1):
        field_name = f"bid_{bid_id}"
        field_definitions[field_name] = (
            RelevanceLabel,
            Field(default=default_level, description=f"Relevance level for bid {bid_id}"),
        )

    # Create base class with the method
    class ExplicitBidRelevanceBase(BaseModel):
        def to_labels(self) -> RelevanceLabels:
            high = []
            medium = []
            low = []

            for field_name, value in self.model_dump().items():
                if field_name.startswith("bid_"):
                    bid_id = int(field_name.split("_")[1])
                    if value == "high":
                        high.append(bid_id)
                    elif value == "medium":
                        medium.append(bid_id)
                    elif value == "low":
                        low.append(bid_id)

            return RelevanceLabels(high=high, medium=medium, low=low)

    ExplicitBidRelevanceMap = create_model(
        "ExplicitBidRelevanceMap", **field_definitions, __base__=ExplicitBidRelevanceBase
    )

    return ExplicitBidRelevanceMap


class BidRelevanceItem(BaseModel):
    bid: str = Field(..., description="The bid ID")
    level: Literal["high", "medium", "low"] = Field(..., description="The relevance level for this bid")


class BidRelevanceResponse(BaseModel):
    mapping: List[BidRelevanceItem] = Field(
        ...,
        description="List of bid IDs and their relevance levels",
    )

    def to_labels(self) -> RelevanceLabels:
        high = [int(item.bid) for item in self.mapping if item.level == "high"]
        medium = [int(item.bid) for item in self.mapping if item.level == "medium"]
        low = [int(item.bid) for item in self.mapping if item.level == "low"]
        return RelevanceLabels(high=high, medium=medium, low=low)


class DomMetadata(BaseModel):
    elements: List[ElementMetadata] = Field(..., description="List of elements in the hierarchy")


class ContentOwners(BaseModel):
    developer: List[int] = Field(
        default_factory=list,
        description="List of block IDs owned by the developer",
    )
    user: List[int] = Field(
        default_factory=list,
        description="List of block IDs owned by users",
    )
    third_party: List[int] = Field(
        default_factory=list,
        description="List of block IDs owned by third parties",
    )


class IntegritySet(BaseModel):
    """Subset of integrity levels allowed in a security policy."""

    levels: List[IntegrityLabel] = Field(
        default_factory=list,
        description="List of integrity levels permitted in the policy",
    )

    @computed_field
    def integrity_levels(self) -> List[str]:
        return [level.value for level in self.levels]


class SecurityPolicy(BaseModel):
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why these content types are needed",
    )
    integrity_levels: IntegritySet = Field(..., description="Subset of integrity levels permitted")
    relevance_level: RelevanceLabel = Field(..., description="Lowest relevance permitted")
