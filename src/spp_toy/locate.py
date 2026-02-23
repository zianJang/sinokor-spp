from typing import Dict, List, Optional, Tuple

from src.spp_toy.constraint import (
    can_place_20ft,
    can_place_20ft_weight,
    can_place_40ft,
    can_place_40ft_weight,
    can_place_structurally,
)
from src.spp_toy.env import BayPair, ContainerSpec, Side
from src.spp_toy.utils import mismatch_cells, pop_next_20ft, priority_key


def place_20ft(baypair: BayPair, side: Side, block_id: int, col: int, tier: int, c: ContainerSpec):
    if c.size_ft != 20:
        raise ValueError("place_20ft called with non-20ft container")

    bay = baypair.get_bay(side)
    bay.occupancy[block_id][(col, tier)]["container"] = c
    baypair.update_summary(side, block_id, col, c)


def place_40ft(baypair: BayPair, block_id: int, col: int, tier: int, c: ContainerSpec):
    if c.size_ft != 40:
        raise ValueError("Place_40ft called with non-40ft container")

    baypair.left.occupancy[block_id][(col, tier)]["container"] = c
    baypair.right.occupancy[block_id][(col, tier)]["container"] = c
    baypair.update_summary(None, block_id, col, c)


# ----------------------------------------------------------------------
# 4. 첫 번째 가능한 슬롯 찾기
# ----------------------------------------------------------------------
# Both structural and weight-wise slot search
def find_first_slot_20ft(
    baypair: BayPair, side: Side, block_id: int, c: ContainerSpec
) -> Optional[Tuple[int, int]]:
    bay = baypair.get_bay(side)
    C = bay.block_total_cols[block_id]
    T = bay.block_tiers[block_id]

    for t in range(T):
        for col in range(C):
            if can_place_20ft(baypair, side, block_id, col, t, c) and can_place_20ft_weight(
                baypair, side, block_id, col, t, c
            ):
                return (col, t)
    return None


def find_first_slot_40ft(
    baypair: BayPair, block_id: int, c: ContainerSpec
) -> Optional[Tuple[int, int]]:
    C = baypair.left.block_total_cols[block_id]
    T = baypair.left.block_tiers[block_id]

    for t in range(T):
        for col in range(C):
            if can_place_40ft(baypair, block_id, col, t, c) and can_place_40ft_weight(
                baypair, block_id, col, t, c
            ):
                return (col, t)
    return None


def stow_by_block_mismatch_then_paired(
    baypair: BayPair,
    bay29_block_lists: Dict[int, int],
    bay31_block_lists: Dict[int, int],
    blocks: List[int] = [1, 2, 3, 4, 5, 6],
) -> None:
    for block_id in blocks:
        left_containers = bay29_block_lists.get(block_id, [])
        right_containers = bay31_block_lists.get(block_id, [])

        # indicator → ContainerSpec 매핑
        left_dict = {c.indicator: c for c in left_containers}
        right_dict = {c.indicator: c for c in right_containers}

        paired_indicators = sorted(set(left_dict) & set(right_dict))
        left_only_indicators = sorted(set(left_dict) - set(right_dict))
        right_only_indicators = sorted(set(right_dict) - set(left_dict))

        reefer_paired_tasks = []
        reefer_left_tasks = []
        reefer_right_tasks = []
        general_paired_tasks = []
        general_left_tasks = []
        general_right_tasks = []

        # 1. Paired (동일 indicator → 반드시 같은 col,tier에 40ft 배치)
        for ind in paired_indicators:
            c_left = left_dict[ind]
            c_right = right_dict[ind]

            # 동일 spec 검증
            if (
                c_left.size_ft != c_right.size_ft
                or c_left.is_reefer != c_right.is_reefer
                or c_left.weight != c_right.weight
            ):
                raise ValueError(
                    f"Indicator {ind} has mismatched specs in block {block_id} "
                    f"(left: {c_left}, right: {c_right})"
                )

            # paired는 반드시 40ft여야 함 (현재 로직 기준)
            if c_left.size_ft != 40:
                raise ValueError(
                    f"Paired containers (indicator {ind}) in block {block_id} "
                    "must be 40ft containers"
                )

            # paired_tasks.append(("paired", c_left))  # c_left나 c_right 둘 중 하나면 충분
            if c_left.is_reefer and c_left.weight == "full":
                reefer_paired_tasks.append(c_left)
            else:
                general_paired_tasks.append(c_left)

        # 2. Left-only / Right-only → 20ft만 허용
        for ind in left_only_indicators:
            c = left_dict[ind]
            if c.size_ft != 20:
                raise ValueError(
                    f"Left-only container (indicator {ind}) in block {block_id} must be 20ft"
                )
            # left_tasks.append(("left", c))
            if c.is_reefer and c.weight == "full":
                reefer_left_tasks.append(c)
            else:
                general_left_tasks.append(c)

        for ind in right_only_indicators:
            c = right_dict[ind]
            if c.size_ft != 20:
                raise ValueError(
                    f"Right-only container (indicator {ind}) in block {block_id} must be 20ft"
                )
            # right_tasks.append(("right", c))
            if c.is_reefer and c.weight == "full":
                reefer_right_tasks.append(c)
            else:
                general_right_tasks.append(c)

        # 우선순위 정렬 (spec 기준, kind 무관)
        reefer_paired_tasks.sort(key=priority_key)
        reefer_left_tasks.sort(key=priority_key)
        reefer_right_tasks.sort(key=priority_key)
        general_paired_tasks.sort(key=priority_key)
        general_left_tasks.sort(key=priority_key)
        general_right_tasks.sort(key=priority_key)

        mism_cells = mismatch_cells(baypair, block_id)
        mism_cells.sort(key=lambda x: (x[2], x[1]))
        placed_indicators: set = set()

        # mismatch cell stowage  #can_place_structurally
        for side, col, tier, reefer in mism_cells:
            if side == 0:
                if reefer == 1:
                    container = pop_next_20ft(reefer_left_tasks)
                else:
                    container = pop_next_20ft(general_left_tasks)

                if container is None:
                    continue
                if can_place_structurally(baypair.left, block_id, col, tier, container):
                    place_20ft(baypair, "left", block_id, col, tier, container)
                    placed_indicators.add(container.indicator)
            else:
                if reefer == 1:
                    container = pop_next_20ft(reefer_right_tasks)
                else:
                    container = pop_next_20ft(general_right_tasks)

                if container is None:
                    continue
                if can_place_structurally(baypair.right, block_id, col, tier, container):
                    place_20ft(baypair, "right", block_id, col, tier, container)
                    placed_indicators.add(container.indicator)

        rest_reefer_left_tasks = [
            c for c in reefer_left_tasks if c.indicator not in placed_indicators
        ]
        rest_reefer_right_tasks = [
            c for c in reefer_right_tasks if c.indicator not in placed_indicators
        ]
        rest_general_left_tasks = [
            c for c in general_left_tasks if c.indicator not in placed_indicators
        ]
        rest_general_right_tasks = [
            c for c in general_right_tasks if c.indicator not in placed_indicators
        ]

        for container in rest_reefer_left_tasks:
            slot = None
            slot = find_first_slot_20ft(baypair, "left", block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible left (20ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_20ft(baypair, "left", block_id, col, tier, container)

        for container in rest_reefer_right_tasks:
            slot = None
            slot = find_first_slot_20ft(baypair, "right", block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible left (20ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_20ft(baypair, "right", block_id, col, tier, container)

        for container in reefer_paired_tasks:
            slot = None
            slot = find_first_slot_40ft(baypair, block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible paired (40ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_40ft(baypair, block_id, col, tier, container)

        for container in rest_general_left_tasks:
            slot = None
            slot = find_first_slot_20ft(baypair, "left", block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible left (20ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_20ft(baypair, "left", block_id, col, tier, container)

        for container in rest_general_right_tasks:
            slot = None
            slot = find_first_slot_20ft(baypair, "right", block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible left (20ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_20ft(baypair, "right", block_id, col, tier, container)

        for container in general_paired_tasks:
            slot = None
            slot = find_first_slot_40ft(baypair, block_id, container)
            if slot is None:
                raise RuntimeError(
                    f"[block {block_id}] No feasible paired (40ft) slot for {container.indicator}"
                )
            col, tier = slot
            place_40ft(baypair, block_id, col, tier, container)
