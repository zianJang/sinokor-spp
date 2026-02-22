"""
env.py
- Domain/state objects for the stowage planning problem (SPP).

Contains:
- ContainerSpec
- Bay
- BayPair

Design note:
- Hard constraints / feasibility checks are intentionally NOT implemented here.
  Put those in constraint.py to keep state & rules decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Set, Tuple

Slot = Tuple[int, int]  # (col, tier)
Side = Literal["left", "right"]


@dataclass(frozen=True)
class ContainerSpec:
    indicator: str
    size_ft: int  # 20 or 40
    is_reefer: bool  # True/False
    weight: str  # "full" or "empty"

    @property
    def weight_value(self) -> int:
        """
        Discrete weight encoding used throughout the repo.

        20ft empty  -> 1
        40ft empty  -> 2
        20ft full   -> 4
        40ft full   -> 8
        """
        if self.size_ft == 20 and self.weight == "empty":
            return 1
        elif self.size_ft == 40 and self.weight == "empty":
            return 2
        elif self.size_ft == 20 and self.weight == "full":
            return 4
        elif self.size_ft == 40 and self.weight == "full":
            return 8
        else:
            raise ValueError("Invalid weight specification")


class Bay:
    """
    A single bay (left or right).
    Holds occupancy state and provides basic derived quantities (e.g., heel).
    """

    def __init__(
        self,
        name: str,
        block_total_cols: Dict[int, int],  # block -> total cols (e.g., 9)
        block_tiers: Dict[int, int],  # block -> num tiers
        hold_valid_cols_by_tier: Dict[int, Dict[int, Set[int]]],  # block -> tier -> valid cols
        hold_blocks: Set[int] = set(),
        deck_blocks: Set[int] = set(),
        port_blocks: Set[int] = set(),
        center_blocks: Set[int] = set(),
        starboard_blocks: Set[int] = set(),
        deck_tier_policy: str = "top",
    ):
        self.name = name
        self.block_total_cols = block_total_cols
        self.block_tiers = block_tiers
        self.hold_valid_cols_by_tier = hold_valid_cols_by_tier
        self.hold_blocks = set(hold_blocks)
        self.deck_blocks = set(deck_blocks)
        self.port_blocks = set(port_blocks)
        self.starboard_blocks = set(starboard_blocks)
        self.center_blocks = set(center_blocks)

        # deck tier per block
        self.deck_tier: Dict[int, int] = {}
        for b, T in self.block_tiers.items():
            self.deck_tier[b] = (T - 1) if deck_tier_policy == "top" else 0

        # occupancy[block][(col,tier)] = {"container": ContainerSpec|None, "valid": 0/1, "reefer": 0/1}
        self.occupancy: Dict[int, Dict[Slot, Dict[str, object]]] = {}
        for b in self.block_total_cols:
            C = self.block_total_cols[b]
            T = self.block_tiers[b]
            self.occupancy[b] = {}
            for t in range(T):
                for c in range(C):
                    if b in self.hold_valid_cols_by_tier and t in self.hold_valid_cols_by_tier[b]:
                        v = 1 if c in self.hold_valid_cols_by_tier[b][t] else 0
                    else:
                        v = 1
                    r = 1 if (b in self.deck_blocks and t == 0) else 0
                    self.occupancy[b][(c, t)] = {"container": None, "valid": v, "reefer": r}

        # Optional: auto-classify hold/deck blocks into port/center/starboard by index ordering
        center_hold_id = len(self.hold_blocks) // 2
        for hold_id in self.hold_blocks:
            if hold_id < center_hold_id:
                self.port_blocks.add(hold_id)
            elif hold_id > center_hold_id:
                self.starboard_blocks.add(hold_id)
            else:
                self.center_blocks.add(hold_id)

        center_deck_id = len(self.deck_blocks) // 2
        for deck_id in self.deck_blocks:
            if deck_id < center_deck_id:
                self.port_blocks.add(deck_id)
            elif deck_id > center_deck_id:
                self.starboard_blocks.add(deck_id)
            else:
                self.center_blocks.add(deck_id)

    def reset_occupancy(
        self,
    ):
        for block_id, slots in self.occupancy.items():
            for key, cell in slots.items():
                cell["container"] = None

    def block_port_starboard_weight(self, block_id: int) -> Tuple[float, float]:
        """
        Compute port/starboard weighted moments within a block
        using distance-to-center * container.weight_value.
        """
        w_port = 0.0
        w_starboard = 0.0
        C = self.block_total_cols[block_id]
        center = (C - 1) / 2  # handles odd/even col counts (distance may be fractional)

        for (col, _tier), cell in self.occupancy[block_id].items():
            container = cell.get("container")
            if not isinstance(container, ContainerSpec):
                continue

            d = col - center
            if d < 0:
                w_port += container.weight_value * abs(d)
            elif d > 0:
                w_starboard += container.weight_value * abs(d)
            # d == 0 contributes 0

        return (w_port, w_starboard)

    def block_heel(self, block_id: int) -> float:
        """Absolute difference between port and starboard weighted moments."""
        w_port, w_starboard = self.block_port_starboard_weight(block_id)
        return abs(w_port - w_starboard)

    def all_blocks_heel(self) -> Dict[int, float]:
        """Heel per block."""
        return {b: self.block_heel(b) for b in self.block_total_cols}


class BayPair:
    """
    Paired bays (left/right) that share the same block structure.

    column_summary_pair[b][col] keeps pair-wise statistics:
      w20_left, w20_right, w40_pair,
      n20_left, n20_right, n40_pair,
      wTot_pair (= w20_left + w20_right + w40_pair)
    """

    def __init__(self, left: Bay, right: Bay):
        self.left = left
        self.right = right

        # paired blocks (left, right) must be homogeneous
        self.block_total_cols = self.left.block_total_cols

        self.column_summary_pair: Dict[int, Dict[int, Dict[str, int]]] = {}
        self.reset_summary()

    def reset(self) -> None:
        """
        Reset paired bay state:
        - Clear all containers in left/right bays
        - Reset column summary statistics
        """
        self.left.reset_occupancy()
        self.right.reset_occupancy()
        self.reset_summary()

    def get_bay(self, side: Side) -> Bay:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError("side must be 'left' or 'right'")

    def reset_summary(self) -> None:
        for b, C in self.block_total_cols.items():
            self.column_summary_pair[b] = {}
            for col in range(C):
                self.column_summary_pair[b][col] = {
                    "w20_left": 0,
                    "w20_right": 0,
                    "w40_pair": 0,
                    "n20_left": 0,
                    "n20_right": 0,
                    "n40_pair": 0,
                    "wTot_pair": 0,
                }

    def update_summary_from_occupancy(self) -> None:
        """
        Recompute column_summary_pair from current occupancy in both bays.
        Useful after bulk edits / debugging.
        """
        self.reset_summary()

        # PASS 1: 20ft only
        for block_id, slots in self.left.occupancy.items():
            for (col, _tier), cell in slots.items():
                c = cell.get("container")
                if not isinstance(c, ContainerSpec) or c.size_ft != 20:
                    continue
                s = self.column_summary_pair[block_id][col]
                s["w20_left"] += c.weight_value
                s["n20_left"] += 1

        for block_id, slots in self.right.occupancy.items():
            for (col, _tier), cell in slots.items():
                c = cell.get("container")
                if not isinstance(c, ContainerSpec) or c.size_ft != 20:
                    continue
                s = self.column_summary_pair[block_id][col]
                s["w20_right"] += c.weight_value
                s["n20_right"] += 1

        # PASS 2: 40ft only (pair-level, dedup by object id)
        seen_40: Set[int] = set()
        for bay in (self.left, self.right):
            for block_id, slots in bay.occupancy.items():
                for (col, _tier), cell in slots.items():
                    c = cell.get("container")
                    if not isinstance(c, ContainerSpec) or c.size_ft != 40:
                        continue
                    cid = id(c)
                    if cid in seen_40:
                        continue
                    seen_40.add(cid)

                    s = self.column_summary_pair[block_id][col]
                    s["w40_pair"] += c.weight_value
                    s["n40_pair"] += 1

        # PASS 3: totals
        for block_id in self.column_summary_pair:
            for col in self.column_summary_pair[block_id]:
                s = self.column_summary_pair[block_id][col]
                s["wTot_pair"] = s["w20_left"] + s["w20_right"] + s["w40_pair"]

    def update_summary(
        self, side: Optional[Side], block_id: int, col: int, c: ContainerSpec
    ) -> None:
        """
        Incremental summary update for a single placed container.
        - 20ft requires explicit side.
        - 40ft updates pair-level summary (side=None).
        """
        summary = self.column_summary_pair[block_id][col]

        if c.size_ft == 20:
            if side is None:
                raise ValueError("20ft requires explicit side")
            if side == "left":
                summary["w20_left"] += c.weight_value
                summary["n20_left"] += 1
            else:
                summary["w20_right"] += c.weight_value
                summary["n20_right"] += 1
        else:
            summary["w40_pair"] += c.weight_value
            summary["n40_pair"] += 1

        summary["wTot_pair"] = summary["w20_left"] + summary["w20_right"] + summary["w40_pair"]
