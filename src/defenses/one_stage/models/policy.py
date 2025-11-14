from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class ElementMetadata(BaseModel):
    bids: List[int] = Field(..., description="Numeric block identifier")
    context: str = Field(..., description="Where this element lives or applies")
    purpose: str = Field(..., description="The intention or role of this element")


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


class AllowedOwners(BaseModel):
    """Owners allowed for a user request."""

    owners: List[Literal["developer", "user", "third_party"]] = Field(
        default_factory=list,
        description="List of allowed owner categories",
    )


class Decision(BaseModel):
    bid: int = Field(..., description="Unique block identifier")
    rationale: str = Field(
        ...,
        description="Rationale for allowing the summary",
    )


class AllowedBids(BaseModel):
    bids: List[Decision] = Field(
        default_factory=list,
        description="List of allowed bids for which summaries are permitted",
    )

    @computed_field
    def allowed_bids(self) -> List[int]:
        return [decision.bid for decision in self.bids]


class BidDecision(BaseModel):
    bid: int = Field(..., description="Unique block identifier")
    decision: Literal["allow", "deny", "unknown"] = Field(..., description="allow, deny, or unknown")
    rationale: Optional[str] = Field(None, description="Brief rationale")


class FirstPassResponse(BaseModel):
    decisions: List[BidDecision] = Field(..., description="Preliminary decisions")

    @computed_field
    def allowed_bids(self) -> List[int]:
        return [d.bid for d in self.decisions if d.decision == "allow"]

    @computed_field
    def unknown_bids(self) -> List[int]:
        return [d.bid for d in self.decisions if d.decision == "unknown"]


class BidSummary(BaseModel):
    bids: List[int] = Field(..., description="List of unique block identifiers")
    summaries: List[str] = Field(..., description="List of concise summaries for each element")


class StrictPassResponse(BaseModel):
    decisions: List[BidDecision] = Field(..., description="Final allow/deny")

    @computed_field
    def allowed_bids(self) -> List[int]:
        return [d.bid for d in self.decisions if d.decision == "allow"]
