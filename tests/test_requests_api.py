# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``requests`` block: parsing, refusals, and the distribution plan its layout lowers to.

The layout ops themselves (registry, inverses, lowering) are relayout's (``relayout.ops``, tested there);
here they are exercised through the block an agent sends.
"""

import pytest

from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.requests_api import Location, Requests, precision_refusal
from hpcagent_bench.precision import Precision
from relayout.ops import Zip


def distribute(digit: int, grid_dim: int = 0) -> dict[str, object]:
    return {"op": "distribute", "digit": digit, "grid_dim": grid_dim}


def block(dim: int, factor: int | str) -> dict[str, object]:
    return {"op": "block", "dim": dim, "factor": factor}


def test_zip_and_unzip_are_listed_under_their_own_keys() -> None:
    raw = {"zip": [{"name": "uv", "fields": ["u", "v"]}], "unzip": [{"name": "z", "fields": ["z_re", "z_im"]}]}
    requests = Requests.from_obj(raw)
    assert [group.op for group in requests.groups] == ["unzip", "zip"]
    assert Zip(name="uv", fields=("u", "v")) in requests.groups  # type: ignore[arg-type]
    assert requests.to_json() == raw


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
        ({"layout": {"A": [{"op": "bogus"}]}}, r"requests.layout\['A'\]\[0\].op 'bogus' is not a layout op"),
        ({"unzip": [{"name": "z", "fields": ["re"]}]}, "two or more arrays"),
        ({"zip": [{"name": "uv", "fields": ["u", "u"]}]}, "names an array twice"),
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
    test_zip_and_unzip_are_listed_under_their_own_keys()
    test_the_block_round_trips_through_json()
    test_a_malformed_block_is_a_request_fault({"grid": []}, "non-empty list of positive ints")
    test_a_malformed_block_is_a_request_fault({"distribution": {}}, "unknown field")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("libraries")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("distribution")
    test_a_field_that_moved_under_requests_is_refused_with_its_new_spelling("sparse_config")
    test_the_plan_replicates_an_empty_op_list_and_carries_location()
    test_with_grid_keeps_the_ops()
    test_precision_is_refused_where_it_changes_nothing_or_names_no_floating_array()
