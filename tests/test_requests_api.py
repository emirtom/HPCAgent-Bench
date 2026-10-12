# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``requests`` block: parsing, the layout op algebra and its lowering to the scatter descriptor.

A layout is an op list over an array's digit tuple; the judge lowers it to the per-axis scheme it
scatters with and refuses, by name, a sequence it cannot lower. The kernel's default distribution
is shown to agents as an op list (``ops_for_axes``), so lowering that list must give the default back.
"""

import itertools

import pytest

from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.requests_api import (
    Block,
    Distribute,
    Location,
    Pad,
    Permute,
    Requests,
    Shuffle,
    Split,
    Unblock,
    inverse_sequence,
    lower_layout,
    ops_for_axes,
    parse_op,
    precision_refusal,
)
from hpcagent_bench.precision import Precision

#: Every split the descriptor scatters with, over a 2-D grid, for 1-3 dimensional arrays.
AXES = (None, (0, "block", 1), (0, "cyclic", 1), (0, "block_cyclic", 4), (1, "block", 1), (1, "block_cyclic", 3))


def distribute(digit: int, grid_dim: int = 0) -> dict[str, object]:
    return {"op": "distribute", "digit": digit, "grid_dim": grid_dim}


def block(dim: int, factor: int | str) -> dict[str, object]:
    return {"op": "block", "dim": dim, "factor": factor}


def lowered(ops: list[dict[str, object]], ndim: int, grid: list[int]) -> list[Split | None]:
    return lower_layout([parse_op(op, "t") for op in ops], ndim, grid, "t")


def test_the_three_schemes_lower_from_their_op_sequences() -> None:
    assert lowered([distribute(0)], 2, [4]) == [(0, "block", 1), None]
    assert lowered([block(1, "P0"), distribute(2)], 2, [4]) == [None, (0, "cyclic", 1)]
    blocked = [block(0, "8*P0"), block(0, 8), {"op": "permute", "perm": [0, 2, 3, 1]}, distribute(1)]
    assert lowered(blocked, 2, [4]) == [(0, "block_cyclic", 8), None]


def test_a_grid_factor_re_evaluates_at_every_rank_count() -> None:
    """One request stays valid when a scaling run re-sizes the grid: ``P0`` is read at each P."""
    ops = [block(0, "2*P0"), block(0, 2), distribute(1)]
    for p in (1, 2, 4, 8):
        assert lowered(ops, 1, [p]) == [(0, "block_cyclic", 2)]


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_default_layout_raises_to_ops_that_lower_back(ndim: int) -> None:
    """``ops_for_axes`` is what an agent is shown; lowering it must be the descriptor it came from."""
    for axes in itertools.product(AXES, repeat=ndim):
        grid_dims = [axis[0] for axis in axes if axis is not None]
        if len(grid_dims) != len(set(grid_dims)):
            continue
        ops = ops_for_axes(list(axes))
        for grid in ([1, 1], [2, 3], [4, 4]):
            expected = [None if a is None else (a[0], a[1], 1 if a[1] in ("block", "cyclic") else a[2]) for a in axes]
            assert lower_layout(ops, ndim, grid, "t") == expected, (axes, grid)


def test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it() -> None:
    ops = [block(0, "4*P0"), block(0, 4), distribute(2)]  # dimension 1's digit sits between dimension 0's
    with pytest.raises(ValueError, match=r'append \{"op": "permute", "perm": \[0, 2, 1\]\}'):
        lowered(ops, 2, [4])


@pytest.mark.parametrize(
    ("ops", "match"),
    [
        ([{"op": "pad", "dim": 0, "amount": 4}], "'pad' is not implemented yet"),
        ([{"op": "shuffle", "dim": 0, "f": "i", "f_inv": "i"}], "'shuffle' is not implemented yet"),
        ([block(0, 3), distribute(2)], "must have extent P0 = 4"),
        ([block(0, 8), distribute(0)], "coarse digit of a blocked dimension"),
        ([distribute(0), distribute(0, 1)], "dimension 0 is already distributed"),
        ([block(2, 4)], "dimension 2 does not exist"),
        ([distribute(0, 3)], "grid_dim 3 does not exist"),
        ([{"op": "permute", "perm": [1, 0, 2]}], "permute of 3 positions over a 2-digit tuple"),
    ],
)
def test_a_sequence_that_cannot_lower_is_refused_by_name(ops: list[dict[str, object]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        lowered(ops, 2, [4, 2])


def test_every_op_has_its_inverse() -> None:
    """pad <-> unpad, permute <-> its inverse permutation, block <-> unblock, shuffle f <-> f_inv."""
    ops = (Pad(0, "4"), Permute((2, 0, 1)), Block(1, "P0"), Shuffle(0, "(i + 3) % 8", "(i + 5) % 8"))
    inverted = inverse_sequence(ops)
    assert inverted == (Shuffle(0, "(i + 5) % 8", "(i + 3) % 8"), Unblock(1, "P0"), Permute((1, 2, 0)), Pad(0, "-(4)"))
    assert inverse_sequence(inverted)[1:] == ops[1:]
    with pytest.raises(ValueError, match="undone by the gather"):
        inverse_sequence((Distribute(0, 0),))


def test_unblock_undoes_block_in_the_digit_tuple() -> None:
    ops = [block(0, 4), {"op": "unblock", "dim": 0, "factor": 4}, distribute(0)]
    assert lowered(ops, 1, [2]) == [(0, "block", 1)]


def test_the_block_round_trips_through_json() -> None:
    raw = {
        "libraries": ["blas"],
        "precision": {"A": "fp32", "x": "bf16"},
        "grid": [2, 2],
        "layout": {"A": [block(0, "2*P0"), block(0, 2), {"op": "permute", "perm": [0, 2, 3, 1]}, distribute(1)]},
        "location": {"A": "device"},
        "sparse": {"M": "csc"},
    }
    requests = Requests.from_obj(raw)
    assert requests.precision == {"A": Precision.FP32, "x": Precision.BF16}
    assert requests.location == {"A": Location.DEVICE}
    assert Requests.from_obj(requests.to_json()) == requests
    assert requests.to_json()["layout"]["A"][0] == {"op": "block", "dim": 0, "factor": "2*P0"}
    assert requests.to_json()["layout"]["A"][1] == {"op": "block", "dim": 0, "factor": 2}
    assert Requests.from_obj(None).to_json() == {}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"grid": []}, "non-empty list of positive ints"),
        ({"grid": [0]}, "non-empty list of positive ints"),
        ({"precision": {"A": "fp42"}}, "Unknown precision"),
        ({"location": {"A": "gpu"}}, "must be 'host' or 'device'"),
        ({"layout": {"A": [{"op": "bogus"}]}}, "not a layout op"),
        ({"layout": {"A": [{"op": "permute", "perm": [0, 0]}]}}, "permutation of 0..n-1"),
        ({"layout": {"A": [{"op": "shuffle", "dim": 0, "f": "i"}]}}, "inverse 'f_inv'"),
        ({"zip": [{"name": "uv", "fields": ["u"]}]}, "two or more arrays"),
        ({"distribution": {}}, "unknown field"),
    ],
)
def test_a_malformed_block_is_a_request_fault(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Requests.from_obj(raw)


@pytest.mark.parametrize("field", ["libraries", "distribution", "sparse_config"])
def test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling(field: str) -> None:
    with pytest.raises(ValueError, match=f"moved under 'requests': {field} -> requests"):
        Submission.from_obj({"language": "c", "source": "x", field: []})


def test_the_plan_replicates_an_empty_op_list_and_carries_location() -> None:
    requests = Requests.from_obj({"grid": [4], "layout": {"A": [distribute(0)], "b": []}, "location": {"b": "device"}})
    plan = requests.plan({"A": 2, "b": 1})
    assert plan["arrays"]["b"] == {"replicated": True, "location": "device"}
    assert plan["arrays"]["A"]["axes"][0] == {"grid_dim": 0, "scheme": "block", "block_size": 1}
    with pytest.raises(ValueError, match="unknown array 'Z'"):
        Requests.from_obj({"grid": [4], "layout": {"Z": []}}).plan({"A": 2})
    with pytest.raises(ValueError, match="zip is not implemented yet"):
        Requests.from_obj({"grid": [4], "zip": [{"name": "uv", "fields": ["A", "b"]}]}).plan({"A": 2, "b": 1})


def test_with_grid_keeps_the_ops() -> None:
    requests = Requests.from_obj({"grid": [4], "layout": {"A": [block(0, "P0"), distribute(1)]}})
    regridded = requests.with_grid([8])
    assert (regridded.grid, regridded.layout) == ((8,), requests.layout)
    assert regridded.plan({"A": 1})["grid"] == [8]


def test_precision_is_refused_where_it_changes_nothing_or_names_no_floating_array() -> None:
    problem = {"A": "float64", "out": "float64"}
    assert precision_refusal({"A": Precision.FP32}, problem) is None
    assert "already the array's problem precision" in str(precision_refusal({"A": Precision.FP64}, problem))
    assert "not a floating dense array" in str(precision_refusal({"idx": Precision.FP32}, problem))


if __name__ == "__main__":
    test_the_three_schemes_lower_from_their_op_sequences()
    test_a_grid_factor_re_evaluates_at_every_rank_count()
    test_the_default_layout_raises_to_ops_that_lower_back(1)
    test_the_default_layout_raises_to_ops_that_lower_back(2)
    test_the_default_layout_raises_to_ops_that_lower_back(3)
    test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it()
    test_a_sequence_that_cannot_lower_is_refused_by_name([{"op": "pad", "dim": 0, "amount": 4}], "'pad' is not")
    test_a_sequence_that_cannot_lower_is_refused_by_name([block(0, 3), distribute(2)], "must have extent P0 = 4")
    test_a_sequence_that_cannot_lower_is_refused_by_name([block(2, 4)], "dimension 2 does not exist")
    test_every_op_has_its_inverse()
    test_unblock_undoes_block_in_the_digit_tuple()
    test_the_block_round_trips_through_json()
    test_a_malformed_block_is_a_request_fault({"grid": []}, "non-empty list of positive ints")
    test_a_malformed_block_is_a_request_fault({"layout": {"A": [{"op": "bogus"}]}}, "not a layout op")
    test_a_malformed_block_is_a_request_fault({"distribution": {}}, "unknown field")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("libraries")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("distribution")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("sparse_config")
    test_the_plan_replicates_an_empty_op_list_and_carries_location()
    test_with_grid_keeps_the_ops()
    test_precision_is_refused_where_it_changes_nothing_or_names_no_floating_array()
