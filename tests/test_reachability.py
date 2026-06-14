from __future__ import annotations

from bn_edge_removal.cortical import CorticalModel
from bn_edge_removal.reachability import (
    ReachabilityConfig,
    verify_all_initial_states,
    verify_all_models,
)


def test_verify_all_initial_states_monotone() -> None:
    model = CorticalModel()
    config = ReachabilityConfig(
        max_steps=5,
        constraint_type="monotone",
        enforce_constraints=True,
    )
    result = verify_all_initial_states(model, config)

    assert result.model_name == "cortical"
    assert result.mode == "strict"
    assert result.total_initial_states == 32
    assert 0 <= result.success_count <= 32
    assert len(result.per_state_results) == 32

    # 到達可能な初期状態について、witnessが正しく作成されているか確認
    for item in result.per_state_results:
        if item.reachable:
            assert item.witness is not None
            assert len(item.witness.states) == item.witness.goal_step + 1
            # 最後の状態が目標状態集合に含まれているか確認
            assert tuple(item.witness.states[-1]) in frozenset(
                tuple(s) for s in model.goal_states()
            )
        else:
            assert item.witness is None


def test_verify_all_models_batch() -> None:
    config_by_model = {
        "cortical": ReachabilityConfig(
            max_steps=5,
            constraint_type="monotone",
        ),
        "wnt5a": ReachabilityConfig(
            max_steps=5,
            constraint_type="recovery",
            recovery_tau=2,
        ),
    }
    results = verify_all_models(
        config_by_model=config_by_model,
        modes=["strict", "ignore"],
    )

    assert len(results) == 4
    modes = {r.mode for r in results}
    assert modes == {"strict", "ignore"}
    model_names = {r.model_name for r in results}
    assert model_names == {"cortical", "wnt5a"}
