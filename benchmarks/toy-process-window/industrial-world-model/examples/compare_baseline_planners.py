"""Compare simple baseline planners on the toy process-window objective.

Run from this directory with:

    python examples/compare_baseline_planners.py

The script intentionally uses only the synthetic physics proxy so it is fast,
deterministic, and easy for new contributors to modify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imwm.physics import process_physics_proxy, scale_action


@dataclass(frozen=True)
class CandidateResult:
    planner: str
    action: np.ndarray
    quality: float
    defect_risk: float
    objective: float
    feasible: bool
    evaluated: int


def score_action(action: np.ndarray) -> tuple[float, float, float, bool]:
    """Return quality, defect risk, objective, and feasibility for an action."""
    proxy = process_physics_proxy(action)
    quality = proxy["quality_score"]
    defect_risk = proxy["defect_risk"]
    objective = quality - 0.55 * defect_risk
    feasible = defect_risk <= 0.20 and quality >= 0.70
    return quality, defect_risk, objective, feasible


def best_candidate(planner: str, actions: np.ndarray) -> CandidateResult:
    """Evaluate candidate actions and keep the best objective."""
    best: CandidateResult | None = None
    feasible_count = 0

    for action in actions:
        quality, defect_risk, objective, feasible = score_action(action)
        feasible_count += int(feasible)
        if best is None or objective > best.objective:
            best = CandidateResult(
                planner=planner,
                action=action,
                quality=quality,
                defect_risk=defect_risk,
                objective=objective,
                feasible=feasible,
                evaluated=len(actions),
            )

    assert best is not None
    return CandidateResult(
        planner=best.planner,
        action=best.action,
        quality=best.quality,
        defect_risk=best.defect_risk,
        objective=best.objective,
        feasible=feasible_count > 0,
        evaluated=best.evaluated,
    )


def random_planner(seed: int = 7, samples: int = 200) -> CandidateResult:
    """Sample random process settings and keep the best candidate."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(0.0, 1.0, size=(samples, 2)).astype(np.float32)
    return best_candidate("random", actions)


def grid_search_planner(points_per_axis: int = 21) -> CandidateResult:
    """Evaluate a small regular grid and keep the best candidate."""
    axis = np.linspace(0.0, 1.0, points_per_axis, dtype=np.float32)
    actions = np.array([(x, y) for x in axis for y in axis], dtype=np.float32)
    return best_candidate("grid_search", actions)


def print_result(result: CandidateResult) -> None:
    physical = scale_action(result.action)
    print(
        f"{result.planner:<12} | evaluated={result.evaluated:>3} "
        f"objective={result.objective:.3f} quality={result.quality:.3f} "
        f"defect_risk={result.defect_risk:.3f} feasible={str(result.feasible):<5} "
        f"laser_power={physical['laser_power']:.1f} scan_speed={physical['scan_speed']:.1f}"
    )


def main() -> None:
    print("Planner      | comparison metrics")
    print("-" * 120)
    for result in (random_planner(), grid_search_planner()):
        print_result(result)


if __name__ == "__main__":
    main()
