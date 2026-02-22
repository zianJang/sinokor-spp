"""
constraint.py
- Feasibility / rule checks for SPP.

Contains:
- Weight limit constants
- Column constraint logic (20/40/mixed)
- Structural placement checks
- Weight-aware placement checks (simulate-then-rollback)

Depends only on env.py.
"""

from __future__ import annotations

from src.spp_toy.env import Bay, BayPair, ContainerSpec, Side

# --------------------------
# Weight limit constants
# --------------------------
MAX_STACK_WEIGHT_COL_PURE_20 = 10  # max weight for pure 20ft = 90t
MAX_STACK_WEIGHT_COL_PURE_40 = 16  # max weight for pure 40ft = 180t
MAX_STACK_WEIGHT_COL_MIX_20 = 8  # max weight for 20ft only in mixed col = 72t
MAX_STACK_WEIGHT_COL_MIX = 16  # max weight for 20+40 in mixed col = 180t
MAX_STACK_WEIGHT_BAYPAIR = 24  # <= 230t (pair column total)


# --------------------------
# Column constraint helpers
# --------------------------
def column_type(baypair: BayPair, block_id: int, col: int) -> str:
    s = baypair.column_summary_pair[block_id][col]
    total_n20 = s["n20_left"] + s["n20_right"]
    total_n40 = s["n40_pair"]

    if total_n20 > 0 and total_n40 > 0:
        return "mixed"
    elif total_n20 > 0:
        return "pure20"
    elif total_n40 > 0:
        return "pure40"
    else:
        return "empty"


def check_column_constraint(baypair: BayPair, block_id: int, col: int) -> bool:
    """
    Column-level weight constraints.

    - pure20:
        w20_left, w20_right <= MAX_STACK_WEIGHT_COL_PURE_20
        wTot_pair <= MAX_STACK_WEIGHT_BAYPAIR
    - pure40:
        w40_pair <= MAX_STACK_WEIGHT_COL_PURE_40
        wTot_pair <= MAX_STACK_WEIGHT_BAYPAIR
    - mixed:
        w20_left, w20_right <= MAX_STACK_WEIGHT_COL_MIX_20
        (w20_left + w40_pair) <= MAX_STACK_WEIGHT_COL_MIX
        (w20_right + w40_pair) <= MAX_STACK_WEIGHT_COL_MIX
        wTot_pair <= MAX_STACK_WEIGHT_BAYPAIR
    """
    s = baypair.column_summary_pair[block_id][col]
    ctype = column_type(baypair, block_id, col)

    if ctype == "pure20":
        return (
            s["w20_left"] <= MAX_STACK_WEIGHT_COL_PURE_20
            and s["w20_right"] <= MAX_STACK_WEIGHT_COL_PURE_20
            and s["wTot_pair"] <= MAX_STACK_WEIGHT_BAYPAIR
        )

    if ctype == "pure40":
        return (
            s["w40_pair"] <= MAX_STACK_WEIGHT_COL_PURE_40
            and s["wTot_pair"] <= MAX_STACK_WEIGHT_BAYPAIR
        )

    if ctype == "mixed":
        return (
            s["w20_left"] <= MAX_STACK_WEIGHT_COL_MIX_20
            and s["w20_right"] <= MAX_STACK_WEIGHT_COL_MIX_20
            and (s["w20_left"] + s["w40_pair"]) <= MAX_STACK_WEIGHT_COL_MIX
            and (s["w20_right"] + s["w40_pair"]) <= MAX_STACK_WEIGHT_COL_MIX
            and s["wTot_pair"] <= MAX_STACK_WEIGHT_BAYPAIR
        )

    return True


# ---------------------------------------------------------------------
# 1) Structural / physical placement checks (ignore weight)
# ---------------------------------------------------------------------
def can_place_structurally(
    bay: Bay,
    block_id: int,
    col: int,
    tier: int,
    c: ContainerSpec,
) -> bool:
    """
    Structural/physical/stacking/reefer checks (weight-agnostic).

    Notes (kept identical to notebook logic):
    - If tier > 0: require that the cell below is either invalid (shape gap) OR occupied.
      (i.e., if below is valid but empty, placement is forbidden.)
    - Reefer constraint is applied only for *full* reefer containers.
      Empty reefer is treated like a general container.
    """
    cell = bay.occupancy[block_id].get((col, tier), None)
    if cell is None:
        return False
    if cell["valid"] != 1:
        return False
    if cell["container"] is not None:
        return False

    # stacking: if tier>0, below must be valid & occupied (unless below invalid or missing)
    if tier > 0:
        below = bay.occupancy[block_id].get((col, tier - 1), None)
        if below is None or below["valid"] == 0:
            pass
        else:
            if below["container"] is None:
                return False

    # reefer: only full reefer must be on reefer socket
    if c.is_reefer and c.weight == "full":
        return cell.get("reefer", 0) == 1

    return True


def can_place_20ft(
    baypair: BayPair,
    side: Side,
    block_id: int,
    col: int,
    tier: int,
    c: ContainerSpec,
) -> bool:
    if c.size_ft != 20:
        return False
    bay = baypair.get_bay(side)
    return can_place_structurally(bay, block_id, col, tier, c)


def can_place_40ft(
    baypair: BayPair,
    block_id: int,
    col: int,
    tier: int,
    c: ContainerSpec,
) -> bool:
    if c.size_ft != 40:
        return False

    # 40ft must satisfy structural checks on BOTH sides at the same (block,col,tier)
    return can_place_structurally(baypair.left, block_id, col, tier, c) and can_place_structurally(
        baypair.right, block_id, col, tier, c
    )


# ---------------------------------------------------------------------
# 2) Weight-aware checks (simulate update -> check -> rollback)
# ---------------------------------------------------------------------
def can_place_20ft_weight(
    baypair: BayPair,
    side: Side,
    block_id: int,
    col: int,
    tier: int,
    c: ContainerSpec,
) -> bool:
    """
    After placing this 20ft, would the column weight constraints still hold?
    """
    if c.size_ft != 20:
        return False

    s = baypair.column_summary_pair[block_id][col]
    backup = s.copy()

    if side == "left":
        s["w20_left"] += c.weight_value
        s["n20_left"] += 1
    else:
        s["w20_right"] += c.weight_value
        s["n20_right"] += 1

    s["wTot_pair"] = s["w20_left"] + s["w20_right"] + s["w40_pair"]

    ok = check_column_constraint(baypair, block_id, col)

    # rollback (restore dict content)
    baypair.column_summary_pair[block_id][col] = backup
    return ok


def can_place_40ft_weight(
    baypair: BayPair,
    block_id: int,
    col: int,
    tier: int,
    c: ContainerSpec,
) -> bool:
    """
    After placing this 40ft (pair placement), would the column weight constraints still hold?
    """
    if c.size_ft != 40:
        return False

    s = baypair.column_summary_pair[block_id][col]
    backup = s.copy()

    s["w40_pair"] += c.weight_value
    s["n40_pair"] += 1
    s["wTot_pair"] = s["w20_left"] + s["w20_right"] + s["w40_pair"]

    ok = check_column_constraint(baypair, block_id, col)

    # rollback
    baypair.column_summary_pair[block_id][col] = backup
    return ok
