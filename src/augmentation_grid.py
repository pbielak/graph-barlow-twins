from typing import Dict, List, Optional

import numpy as np


def make_augmentation_params_grid(
    p_x_min: float, p_x_max: float, p_x_step: float,
    p_e_min: float, p_e_max: float, p_e_step: float,
    grid_type: str,
) -> List[Dict[str, Optional[float]]]:
    all_pairs = [
        (p_x, p_e)
        for p_x in np.arange(p_x_min, p_x_max + p_x_step, p_x_step)
        for p_e in np.arange(p_e_min, p_e_max + p_e_step, p_e_step)
    ]

    if grid_type == "ONLY_SAME":
        return [
            {"p_x_1": p_x, "p_e_1": p_e, "p_x_2": None, "p_e_2": None}
            for p_x, p_e in all_pairs
        ]
    else:
        assert grid_type == "ALL"

        # Build all parameter combinations
        grid = set()

        for p1 in all_pairs:
            for p2 in all_pairs:
                # Filter out symmetric pairs
                if (p1, p2) in grid or (p2, p1) in grid:
                    continue

                grid.add((p1, p2))

        return [
            {"p_x_1": p_x_1, "p_e_1": p_e_1, "p_x_2": p_x_2, "p_e_2": p_e_2}
            for (p_x_1, p_e_1), (p_x_2, p_e_2) in sorted(grid)
        ]


def is_same(augmentation_parameters: Dict[str, Optional[float]]) -> bool:
    return (
               augmentation_parameters["p_x_1"] == augmentation_parameters["p_x_2"]
               and
               augmentation_parameters["p_e_1"] == augmentation_parameters["p_e_2"]
           ) or (
               augmentation_parameters["p_x_2"] is None
               and
               augmentation_parameters["p_e_2"] is None
           )

