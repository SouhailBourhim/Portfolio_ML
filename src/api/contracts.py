"""Typed contracts for the read-only research API.

The API serves published allocations and evidence artifacts.  It never accepts
market data or refits a strategy in a request, so these schemas deliberately
describe a *published research allocation*, not a live trading signal.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SnapshotProvenance(BaseModel):
    """Identity of the release artifact backing an API response.

    `status` carries only states this service can actually emit. An earlier
    draft offered `verified_snapshot`, which nothing produced — the API reads
    the manifest but never recomputes its checksums, so a client branching on
    that value had unreachable code and, worse, could infer the API verifies
    releases. `manifest_dirty_tree` replaces it because that state is real,
    reachable, and dangerous: `snapshot.py verify` REJECTS a manifest written
    from a dirty working tree, since its `producer_commit` does not identify
    the code that produced the artifacts.
    """

    status: Literal["manifest_present", "manifest_dirty_tree", "unavailable"]
    manifest_path: str | None = None
    producer_commit: str | None = None
    producer_tree_dirty: bool | None = None
    generated_at_utc: str | None = None
    files_hashed: int | None = Field(default=None, ge=0)
    note: str


class ApiVersionResponse(BaseModel):
    api_version: str
    service_kind: Literal["read_only_research_artifact_api"]
    research_only: Literal[True]
    order_execution_supported: Literal[False]
    provenance: SnapshotProvenance


class PublishedAllocationResponse(BaseModel):
    """A historic/published allocation, never an on-demand recommendation."""

    contract: Literal["published_research_allocation/v1"]
    universe: str
    strategy: str
    as_of: str
    weights: dict[str, float]
    research_only: Literal[True]
    order_execution_supported: Literal[False]
    provenance: SnapshotProvenance
    # Required, not optional: an allocation quoted without it omits a material
    # economic exposure that every weight in this project carries.
    currency_exposure: str
    caveat: str


class ExplanationResponse(BaseModel):
    """Versioned explanation of the published final-rebalance decision."""

    contract: Literal["published_decision_explanation/v1"]
    universe: str
    decision_date: str
    explanation: dict
    provenance: SnapshotProvenance
    caveat: str


class ModelCardResponse(BaseModel):
    contract: Literal["model_card/v1"]
    system: str
    markdown: str
    provenance: SnapshotProvenance
