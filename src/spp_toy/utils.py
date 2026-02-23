from typing import Tuple

from src.spp_toy.env import BayPair, ContainerSpec


def mismatch_cells(baypair: BayPair, block_id: int):
    C = baypair.left.block_total_cols[block_id]
    T = baypair.left.block_tiers[block_id]
    out = []
    for tier in range(T):
        for col in range(C):
            vL = baypair.left.occupancy[block_id][(col, tier)]["valid"]
            vR = baypair.right.occupancy[block_id][(col, tier)]["valid"]
            eL = baypair.left.occupancy[block_id][(col, tier)]["container"] is None
            eR = baypair.right.occupancy[block_id][(col, tier)]["container"] is None
            rL = baypair.left.occupancy[block_id][(col, tier)]["reefer"]
            rR = baypair.right.occupancy[block_id][(col, tier)]["reefer"]

            if vL == 1 and vR == 0 and eL:
                out.append((0, col, tier, rL))
            elif vL == 0 and vR == 1 and eR:
                out.append((1, col, tier, rR))
            elif vL == 1 and vR == 1 and (eL != eR):
                # side_to_fill = 0 if eL else 1
                if eL:
                    out.append((0, col, tier, rL))
                else:
                    out.append((1, col, tier, rR))
    return out


def pop_next_20ft(tasks):
    for i, c in enumerate(tasks):
        if c.size_ft == 20:
            return tasks.pop(i)
    return None


def priority_key(c: ContainerSpec) -> Tuple[int, str]:
    """
    '20ft reefer full ->
     40ft reefer full ->
     20ft nonreefer full ->
     20ft (non)reefer empty ->
     40ft nonreefer full ->
     40ft (non)reefer empty'

    Reefer full -> General(nonreefer) full -> empty
    """

    if c.weight == "empty":
        p = 2  # G,E / R,E
    else:  # c.weight == "full"
        if c.is_reefer:
            p = 0  # R,F
        else:
            p = 1  # G,F
    # 동일 priority 내에서는 indicator로 안정 정렬
    return (p, c.indicator)
