# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``requests`` block: parsing, refusals, and the distribution plan its layout lowers to.

The layout ops themselves (registry, inverses, lowering) are relayout's (``relayout.ops``, tested there);
here they are exercised through the block an agent sends.
"""

import pytest
from relayout.ops import Unzip, Zip

from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.requests_api import Location, Requests, grid_request, precision_refusal
from hpcagent_bench.precision import Precision


def distribute(digit: str, grid_dim: int = 0) -> dict[str, object]:
    return {"op": "distribute", "digit": digit, "grid_dim": grid_dim}


def split(digit: str, parts: int | str, into: list[str]) -> dict[str, object]:
    return {"op": "split", "digit": digit, "parts": parts, "into": into}


def block(digit: str, factor: int | str, into: list[str]) -> dict[str, object]:
    return {"op": "block", "digit": digit, "factor": factor, "into": into}


#: Dimension 0 in contiguous bands over grid dimension 0.
BANDS = [split("i0", "P0", ["r", "i0"]), distribute("r")]


def test_zip_and_unzip_are_requested_as_name_to_fields() -> None:
    raw = {"zip": {"uv": ["u", "v"]}, "unzip": {"z": ["z_re", "z_im"]}}
    requests = Requests.from_obj(raw)
    assert requests.groups == (Unzip("z", ("z_re", "z_im")), Zip("uv", ("u", "v")))
    assert requests.to_json() == raw


def test_the_block_round_trips_through_json() -> None:
    raw = {
        "libraries": ["blas"],
        "precision": {"A": "fp32", "x": "bf16"},
        "grid": ["root(P, 2)", "root(P, 2)"],
        "layout": {"A": [block("i0", "2*P0", ["i0", "c"]), block("c", 2, ["r", "b"]), distribute("r")]},
        "location": {"A": "device"},
        "sparse": {"M": "csc"},
    }
    requests = Requests.from_obj(raw)
    assert requests.precision == {"A": Precision.FP32, "x": Precision.BF16}
    assert requests.location == {"A": Location.DEVICE}
    assert Requests.from_obj(requests.to_json()) == requests
    assert requests.to_json() == raw
    assert Requests.from_obj(None).to_json() == {}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"grid": []}, "non-empty list of extents over P"),
        ({"grid": ["P ** 2"]}, "uses Pow"),
        ({"precision": {"A": "fp42"}}, "Unknown precision"),
        ({"grid": ["P"], "location": {"A": "gpu"}}, "must be 'host' or 'device'"),
        ({"location": {"A": "device"}}, "needs requests.grid"),
        ({"layout": {"A": [{"op": "bogus"}]}}, r"requests.layout\['A'\]\[0\].op 'bogus' is not a layout op"),
        ({"unzip": {"z": ["re"]}}, "two or more arrays"),
        ({"zip": {"uv": ["u", "u"]}}, "names an array twice"),
        ({"zip": [{"name": "uv", "fields": ["u", "v"]}]}, r"must be an object \{name: \[two or more arrays\]\}"),
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
    requests = Requests.from_obj({"grid": ["P"], "layout": {"A": BANDS, "b": []}, "location": {"b": "device"}})
    plan = requests.plan({"A": 2, "b": 1}, 4)
    assert plan["grid"] == [4]
    assert plan["arrays"]["b"] == {"replicated": True, "location": "device"}
    assert plan["arrays"]["A"]["axes"][0] == {"grid_dim": 0, "scheme": "block", "block_size": 1}
    with pytest.raises(ValueError, match="unknown array 'Z'"):
        Requests.from_obj({"grid": ["P"], "layout": {"Z": []}}).plan({"A": 2}, 4)
    with pytest.raises(ValueError, match="zip is not implemented yet"):
        Requests.from_obj({"grid": ["P"], "zip": {"uv": ["A", "b"]}}).plan({"A": 2, "b": 1}, 4)


def test_the_grid_is_evaluated_at_each_rank_count() -> None:
    """One request is graded at every P a scaling run measures; a P its grid does not span is refused."""
    square = Requests.from_obj({"grid": ["root(P, 2)", "root(P, 2)"], "layout": {"A": BANDS}})
    assert square.plan({"A": 2}, 9)["grid"] == [3, 3]
    assert square.grid_at(8) is None
    with pytest.raises(ValueError, match="does not span 8 rank"):
        square.plan({"A": 2}, 8)
    cyclic = [block("i0", "P0", ["i0", "r"]), distribute("r")]
    with pytest.raises(ValueError, match="needs an equal-edge grid"):
        Requests.from_obj({"grid": ["P // 2", 2], "layout": {"A": cyclic}}).plan({"A": 2}, 8)


def test_a_plan_grid_is_requested_so_that_it_follows_p() -> None:
    """A kernel's default (and a v6 grade's) grid at one rank count becomes the request a scaling run
    evaluates at every P: 1-D follows P, an equal-edge grid keeps its edges equal."""
    assert grid_request([4]) == ("P",)
    assert grid_request([2, 2]) == ("root(P, 2)", "root(P, 2)")
    assert grid_request([2, 4]) == ("2", "4")


def test_precision_is_refused_where_it_changes_nothing_or_names_no_floating_array() -> None:
    problem = {"A": "float64", "out": "float64"}
    assert precision_refusal({"A": Precision.FP32}, problem) is None
    assert "already the array's problem precision" in str(precision_refusal({"A": Precision.FP64}, problem))
    assert "not a floating dense array" in str(precision_refusal({"idx": Precision.FP32}, problem))


if __name__ == "__main__":
    test_zip_and_unzip_are_requested_as_name_to_fields()
    test_the_block_round_trips_through_json()
    test_a_malformed_block_is_a_request_fault({"grid": []}, "non-empty list of extents over P")
    test_a_malformed_block_is_a_request_fault({"location": {"A": "device"}}, "needs requests.grid")
    test_a_malformed_block_is_a_request_fault({"distribution": {}}, "unknown field")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("libraries")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("distribution")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("sparse_config")
    test_the_plan_replicates_an_empty_op_list_and_carries_location()
    test_the_grid_is_evaluated_at_each_rank_count()
    test_a_plan_grid_is_requested_so_that_it_follows_p()
    test_precision_is_refused_where_it_changes_nothing_or_names_no_floating_array()
