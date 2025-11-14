from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import IntegrityLabel


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

    @property
    def allowed_bids(self) -> List[int]:
        return [decision.bid for decision in self.bids]


class IntegritySet(BaseModel):
    """Subset of integrity levels allowed in a security policy."""

    levels: List[IntegrityLabel] = Field(
        default_factory=list,
        description="List of integrity levels permitted in the policy",
    )

    @property
    def integrity_levels(self) -> List[str]:
        return [level.value for level in self.levels]


class SecurityPolicy(BaseModel):
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why these content types are needed",
    )
    integrity_levels: IntegritySet = Field(..., description="Subset of integrity levels permitted")
