import matplotlib.pyplot as plt

# ---------- 색 지정 ----------
# ---------- Cell 색 ----------
COLOR_EMPTY_REEFER_CELL = "#e6f7ff"
COLOR_EMPTY_GENERAL = "#ffffff"
COLOR_INVALID = "#e0e0e0"

# ---------- Container 색 ----------
# Reefer (cool tone)
COLOR_R_20 = "#96cce0"
COLOR_R_40 = "#0096c7"

# General (warm neutral)
COLOR_G_20 = "#acecb5"
COLOR_G_40 = "#10b64a"


def _container_facecolor(c, cell_is_reefer: bool):
    """
    색은 '컨테이너 타입(reefer/nonreefer) & size' 기준으로 결정.
    cell_is_reefer는 라벨링/검증용으로만 두고, 색은 컨테이너 spec으로만 칠함.
    """
    if c.is_reefer:
        return COLOR_R_20 if c.size_ft == 20 else COLOR_R_40
    else:
        return COLOR_G_20 if c.size_ft == 20 else COLOR_G_40


def _container_text(c):
    """
    셀 안에:
      1줄: 고유번호
      2줄: R/G  F/E
    """
    rg = "R" if c.is_reefer else "G"
    fe = "F" if c.weight == "full" else "E"
    return f"{c.indicator}\n{rg}   {fe}"


def _draw_bay_with_containers_on_ax(ax, bay, title: str):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    # block -> global col group index (0,1,2) * 3
    block_to_group = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}
    cell_size = 1.0

    for b in range(1, 7):
        C = bay.block_total_cols[b]
        T = bay.block_tiers[b]
        g0 = block_to_group[b] * 3

        for t in range(T):
            for ccol in range(C):
                cell = bay.occupancy[b][(ccol, t)]
                valid = cell["valid"]
                is_reefer_cell = cell.get("reefer", 0) == 1
                cont = cell.get("container", None)

                # 좌표: deck는 y=4..7, hold는 y=0..3
                y = (4 + t) if (b in {1, 2, 3}) else t
                x = g0 + ccol

                # ----- 배경색 결정 -----
                if valid == 0:
                    face = COLOR_INVALID
                else:
                    if cont is None:
                        face = COLOR_EMPTY_REEFER_CELL if is_reefer_cell else COLOR_EMPTY_GENERAL
                    else:
                        face = _container_facecolor(cont, is_reefer_cell)

                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        cell_size,
                        cell_size,
                        facecolor=face,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )

                # ----- 텍스트 -----
                if valid == 0:
                    ax.text(x + 0.5, y + 0.5, "X", ha="center", va="center", fontsize=9)
                elif cont is not None:
                    ax.text(
                        x + 0.5,
                        y + 0.53,
                        _container_text(cont),
                        ha="center",
                        va="center",
                        fontsize=8,
                        linespacing=1.1,
                        color="black",
                    )

    # separators between block groups
    ax.plot([3, 3], [0, 8], linewidth=2, color="black")
    ax.plot([6, 6], [0, 8], linewidth=2, color="black")

    # deck line
    ax.plot([0, 9], [4, 4], linewidth=2.5, color="black")
    ax.text(-0.6, 6.5, "deck", rotation=90, va="center", fontsize=11)
    ax.text(-0.6, 1.5, "hold", rotation=90, va="center", fontsize=11)

    # col labels
    for gc in range(9):
        ax.text(gc + 0.5, 8.3, str(gc), ha="center", va="bottom", fontsize=10)
    ax.text(4.5, 8.8, "col", ha="center", va="bottom", fontsize=11)

    # tier labels
    for tt in range(4):
        ax.text(-0.2, 4 + tt + 0.5, str(tt), ha="right", va="center", fontsize=10)
        ax.text(-0.2, tt + 0.5, str(tt), ha="right", va="center", fontsize=10)
    ax.text(-1.0, 4.0, "tier", ha="center", va="center", fontsize=11, rotation=90)

    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-0.5, 9.2)


def visualize_bay_with_containers(bay, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    _draw_bay_with_containers_on_ax(ax, bay, title or f"{bay.name} stowage")
    plt.show()


def visualize_baypair_with_containers(
    baypair, pair_name="Bay30", left_name="Bay29", right_name="Bay31"
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(pair_name, fontsize=14)

    _draw_bay_with_containers_on_ax(axes[0], baypair.left, f"{left_name} stowage")
    _draw_bay_with_containers_on_ax(axes[1], baypair.right, f"{right_name} stowage")

    plt.show()
