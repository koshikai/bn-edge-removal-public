"""S. pombe model: source rules, per-edge influence and goal semantics."""

from __future__ import annotations

import re
import warnings
from itertools import product

import pytest

from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
    trajectory_satisfies_goal,
    unmaintainable_goal_states,
)
from bn_edge_removal.models.network_spec import (
    NetworkEdge,
    load_network_spec,
)
from bn_edge_removal.models.s_pombe import SPombeModel

_SPEC = load_network_spec("s_pombe")
_G1 = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
_ALL_STATES = [list(bits) for bits in product([0, 1], repeat=_SPEC.n_nodes)]
_ALL_INPUTS = [list(bits) for bits in product([0, 1], repeat=_SPEC.m_edges)]

# biodivine-boolean-models id-095 FISSION-YEAST-2008, model.bnet (repository commit
# 314b2fe).  v_Start is an input of that model and has no rule; the implementation
# follows the original paper instead, where Start switches off after one step.
_BIODIVINE_ID095_BNET = """
v_Cdc25, ((((!v_Cdc2_Cdc13 & v_Cdc25) & !v_PP) | ((v_Cdc2_Cdc13 & !v_Cdc25) & !v_PP)) | (v_Cdc2_Cdc13 & v_Cdc25))
v_Cdc2_Cdc13, ((!v_Ste9 & !v_Rum1) & !v_Slp1)
v_Cdc2_Cdc13_A, ((((!v_Ste9 & !v_Rum1) & !v_Slp1) & !v_Wee1_Mik1) & v_Cdc25)
v_PP, v_Slp1
v_Rum1, ((((((((!v_SK & !v_Cdc2_Cdc13) & !v_Rum1) & !v_Cdc2_Cdc13_A) & v_PP) | (((!v_SK & !v_Cdc2_Cdc13) & v_Rum1) & !v_Cdc2_Cdc13_A)) | ((((!v_SK & !v_Cdc2_Cdc13) & v_Rum1) & v_Cdc2_Cdc13_A) & v_PP)) | ((((!v_SK & v_Cdc2_Cdc13) & v_Rum1) & !v_Cdc2_Cdc13_A) & v_PP)) | ((((v_SK & !v_Cdc2_Cdc13) & v_Rum1) & !v_Cdc2_Cdc13_A) & v_PP))
v_SK, v_Start
v_Slp1, v_Cdc2_Cdc13_A
v_Ste9, ((((((((!v_SK & !v_Cdc2_Cdc13) & !v_Ste9) & !v_Cdc2_Cdc13_A) & v_PP) | (((!v_SK & !v_Cdc2_Cdc13) & v_Ste9) & !v_Cdc2_Cdc13_A)) | ((((!v_SK & !v_Cdc2_Cdc13) & v_Ste9) & v_Cdc2_Cdc13_A) & v_PP)) | ((((!v_SK & v_Cdc2_Cdc13) & v_Ste9) & !v_Cdc2_Cdc13_A) & v_PP)) | ((((v_SK & !v_Cdc2_Cdc13) & v_Ste9) & !v_Cdc2_Cdc13_A) & v_PP))
v_Wee1_Mik1, ((((!v_Cdc2_Cdc13 & !v_Wee1_Mik1) & v_PP) | (!v_Cdc2_Cdc13 & v_Wee1_Mik1)) | ((v_Cdc2_Cdc13 & v_Wee1_Mik1) & v_PP))
"""  # noqa: E501
_BIODIVINE_NAMES = {
    "v_Start": 0,
    "v_SK": 1,
    "v_Cdc2_Cdc13": 2,
    "v_Cdc2_Cdc13_A": 3,
    "v_Ste9": 4,
    "v_Rum1": 5,
    "v_Slp1": 6,
    "v_PP": 7,
    "v_Cdc25": 8,
    "v_Wee1_Mik1": 9,
}


def _biodivine_rules() -> dict[int, str]:
    rules: dict[int, str] = {}
    for line in _BIODIVINE_ID095_BNET.strip().splitlines():
        target, factors = (part.strip() for part in line.split(",", 1))
        expression = (
            factors.replace("!", " not ").replace("&", " and ").replace("|", " or ")
        )
        rules[_BIODIVINE_NAMES[target]] = re.sub(
            r"v_\w+", lambda m: f"x[{_BIODIVINE_NAMES[m.group(0)]}]", expression
        )
    return rules


def _influence(
    model: SPombeModel, source: int, target: int, u: list[int]
) -> tuple[int, int]:
    """Number of states where raising x_source raises / lowers x_target'."""
    raises = lowers = 0
    for x in _ALL_STATES:
        if x[source - 1]:
            continue
        high = x.copy()
        high[source - 1] = 1
        before = model.next_state(x, u)[target - 1]
        after = model.next_state(high, u)[target - 1]
        raises += after > before
        lowers += after < before
    return raises, lowers


def test_model_dimensions():
    model = SPombeModel()
    assert model.n_nodes == 10
    assert model.m_edges == 4
    assert len(_SPEC.all_edges) == 27


def test_u0_dynamics_match_biodivine_id095_except_start():
    model = SPombeModel()
    rules = _biodivine_rules()
    assert sorted(rules) == list(range(1, 10))

    for x in _ALL_STATES:
        expected = [0] + [
            int(bool(eval(rules[i], {"__builtins__": {}}, {"x": x})))
            for i in range(1, 10)
        ]
        assert model.next_state(x, [0, 0, 0, 0]) == expected, f"x={x}"


@pytest.mark.parametrize(
    "edge", _SPEC.all_edges, ids=lambda e: f"{e.source}to{e.target}-{e.sign}"
)
def test_every_declared_edge_has_the_declared_influence(edge: NetworkEdge):
    """Per-edge check over all 27 edges: sign at u=0, and what removal does."""
    model = SPombeModel()
    raises, lowers = _influence(model, edge.source, edge.target, [0, 0, 0, 0])
    if edge.sign == "activation":
        assert raises > 0 and lowers == 0
    else:
        assert lowers > 0 and raises == 0

    for u in _ALL_INPUTS:
        influence = _influence(model, edge.source, edge.target, u)
        if edge.removable_index is not None and u[edge.removable_index - 1]:
            assert influence == (0, 0), f"u={u}"
        else:
            # Fixed edges, and removable edges that are not removed, keep the sign.
            assert (influence[0] > 0, influence[1] > 0) == (raises > 0, lowers > 0)


def test_node_pairs_outside_all_edges_have_no_influence():
    model = SPombeModel()
    declared = {(edge.source, edge.target) for edge in _SPEC.all_edges}
    for source in range(1, 11):
        for target in range(1, 11):
            if (source, target) in declared:
                continue
            for u in _ALL_INPUTS:
                assert _influence(model, source, target, u) == (0, 0)


def test_pp_does_not_regulate_active_cdc2_cdc13():
    """PP -| Cdc2_Cdc13_star was removable edge u4 until 2026-10-05; not in source."""
    model = SPombeModel()
    assert (8, 4) not in {(edge.source, edge.target) for edge in _SPEC.all_edges}
    assert _influence(model, 8, 4, [0, 0, 0, 0]) == (0, 0)


def test_goal_is_the_g1_fixed_point():
    model = SPombeModel()
    assert model.goal_states() == [_G1]
    assert model.is_goal(_G1)
    assert model.next_state(_G1, [0, 0, 0, 0]) == _G1
    assert unmaintainable_goal_states(model) == []


def test_former_goal_state_cannot_be_maintained():
    """The goal used until 2026-10-05 leaves itself under every control input."""
    model = SPombeModel()
    former_goal = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    assert all(model.next_state(former_goal, u) != former_goal for u in _ALL_INPUTS)


def test_goal_preserving_actions_keep_the_goal():
    model = SPombeModel()
    env = EdgeRemovalEnv(
        model=model,
        reward=RewardConfig(
            step=1.0, terminal=1.0, edge_cost=0.1, violation_penalty=1.0
        ),
        horizon=HorizonConfig(reach_horizon=5, max_steps=10),
    )
    env.reset(initial_state=_G1)

    preserving = env.goal_preserving_actions()

    # Removing both Ste9 -| Cdc2_Cdc13 and Rum1 -| Cdc2_Cdc13 switches Cdc2_Cdc13 on.
    assert preserving == [a for a in range(16) if not (a & 0b1000 and a & 0b0100)]
    assert 0 in preserving


def test_free_dynamics_reach_and_maintain_g1_from_start_off_state():
    model = SPombeModel()
    x = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    trajectory = [x]
    for _ in range(10):
        x = model.next_state(x, [0, 0, 0, 0])
        trajectory.append(x)

    assert trajectory[-1] == _G1
    assert trajectory.index(_G1) == 7
    assert trajectory_satisfies_goal(
        trajectory, model.goal_states(), reach_horizon=7, max_steps=10
    )
    assert not trajectory_satisfies_goal(
        trajectory, model.goal_states(), reach_horizon=6, max_steps=10
    )


def test_start_switches_off_after_one_step_under_every_input():
    model = SPombeModel()
    assert all(model.next_state(x, u)[0] == 0 for x in _ALL_STATES for u in _ALL_INPUTS)


def test_goal_is_reachable_from_every_state_with_control():
    """Held constant (until 2026-10-05), Start = 1 put G1 out of reach."""
    model = SPombeModel()
    reaching = {tuple(_G1)}
    steps = 0
    while True:
        added = {
            tuple(x)
            for x in _ALL_STATES
            if tuple(x) not in reaching
            and any(tuple(model.next_state(x, u)) in reaching for u in _ALL_INPUTS)
        }
        if not added:
            break
        reaching |= added
        steps += 1

    assert len(reaching) == 1024
    assert steps == 9


def test_env_construction_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EdgeRemovalEnv(
            model=SPombeModel(),
            reward=RewardConfig(
                step=1.0, terminal=1.0, edge_cost=0.1, violation_penalty=1.0
            ),
            horizon=HorizonConfig(reach_horizon=5, max_steps=10),
        )
