import os
import pytest
import sqlite3

from coverage import Coverage

from experiment import (
    SUBJECTS_DIR, NON_FLAKY, OD_FLAKY, FLAKY, N_RUNS, update_collated_runs, 
    update_collated_cov, get_req_runs_label_nid, get_features_nid_cov
)


@pytest.fixture
def coverage_data(tmp_path):
    data_file = tmp_path / "cov.sqlite3"
    yield data_file, Coverage(data_file=data_file).get_data()


def test_update_collated_runs():
    collated_proj = [{}, None, None, None]

    update_collated_runs(
        ["passed\ttest1", "passed\ttest2"], "baseline", 0, collated_proj
    )

    assert collated_proj[0]["test1"][0] == {"baseline": [1, 0, None, 0]}
    assert collated_proj[0]["test2"][0] == {"baseline": [1, 0, None, 0]}

    update_collated_runs(
        ["passed\ttest1", "failed\ttest2"], "shuffle", 0, collated_proj
    )

    assert collated_proj[0]["test1"][0] == {
        "baseline": [1, 0, None, 0], "shuffle": [1, 0, None, 0]
    }

    assert collated_proj[0]["test2"][0] == {
        "baseline": [1, 0, None, 0], "shuffle": [1, 1, 0, None]
    }

    update_collated_runs(
        ["failed\ttest1", "passed\ttest2"], "baseline", 1, collated_proj
    )

    assert collated_proj[0]["test1"][0] == {
        "baseline": [2, 1, 1, 0], "shuffle": [1, 0, None, 0]
    }

    assert collated_proj[0]["test2"][0] == {
        "baseline": [2, 0, None, 0], "shuffle": [1, 1, 0, None]
    }

    update_collated_runs(
        ["failed\ttest1", "failed\ttest2"], "shuffle", 1, collated_proj
    )

    assert collated_proj[0]["test1"][0] == {
        "baseline": [2, 1, 1, 0], "shuffle": [2, 1, 1, 0]
    }

    assert collated_proj[0]["test2"][0] == {
        "baseline": [2, 0, None, 0], "shuffle": [2, 2, 0, None]
    }


def test_update_collated_cov(coverage_data):
    proj_dir = os.path.join(SUBJECTS_DIR, "proj", "proj")
    collated_proj = [{}, None, None, None]
    data_file, cov_data = coverage_data
    cov_data.set_context("test1")

    cov_data.add_lines({
        os.path.join(proj_dir, "file1"): {1: None, 2: None}, 
        os.path.join(proj_dir, "file2"): {1: None, 2: None}
    })

    cov_data.set_context("test2")

    cov_data.add_lines({
        os.path.join(proj_dir, "file2"): {2: None, 3: None}, 
        os.path.join(proj_dir, "file3"): {2: None, 3: None}
    })

    with sqlite3.connect(data_file) as con:
        update_collated_cov(con, "proj", collated_proj)

    assert collated_proj[0]["test1"][1] == {
        "file1": {1, 2}, "file2": {1, 2}
    }

    assert collated_proj[0]["test2"][1] == {
        "file2": {2, 3}, "file3": {2, 3}
    }


@pytest.mark.parametrize(
    "runs_nid,expected", 
    [
        (
            {
                "baseline": [N_RUNS["baseline"] - 1, 0, None, 0],
                "shuffle": [N_RUNS["shuffle"] - 1, 0, None, 0]
            },
            (0, None)
        ),
        (
            {
                "baseline": [N_RUNS["baseline"], 0, None, 0],
                "shuffle": [N_RUNS["shuffle"], 0, None, 0]
            },
            (0, NON_FLAKY)
        ),
        (
            {
                "baseline": [N_RUNS["baseline"], 0, None, 0],
                "shuffle": [N_RUNS["shuffle"], 1, 1, 0]
            },
            (1, OD_FLAKY)
        ),
        (
            {
                "baseline": [N_RUNS["baseline"], N_RUNS["baseline"], 0, None],
                "shuffle": [N_RUNS["shuffle"], N_RUNS["shuffle"], 0, None]
            },
            (0, NON_FLAKY)
        ),
        (
            {
                "baseline": [N_RUNS["baseline"], N_RUNS["baseline"], 0, None],
                "shuffle": [N_RUNS["shuffle"], N_RUNS["shuffle"] - 1, None, 1]
            },
            (1, OD_FLAKY)
        ),
        (
            {
                "baseline": [N_RUNS["baseline"], 1, 1, 0],
                "shuffle": [N_RUNS["shuffle"], 0, None, 0]
            },
            (1, FLAKY)
        )
    ]
)
def test_get_req_runs_label_nid(runs_nid, expected):
    assert get_req_runs_label_nid(runs_nid) == expected


@pytest.mark.parametrize(
    "cov_nid,test_files,churn,expected", 
    [
        (
            {"file1.py": {1, 2, 3}, "file2.py": {1, 2, 3}},
            {"file1.py"},
            {"file1.py": {1: 1}, "file2.py": {1: 1, 2: 2}},
            (6, 4, 3)
        ),
        (
            {"file1.py": {1, 2, 3}, "file2.py": {1, 2, 3}},
            set(),
            {"file1.py": {1: 1}, "file2.py": {1: 1, 2: 2}},
            (6, 4, 6)
        ),
        (
            {"file1.py": {1, 2, 3}, "file2.py": {1, 2, 3}},
            set(),
            {"file1.py": {1: 10}, "file2.py": {1: 10, 2: 20}},
            (6, 40, 6)
        ),
    ]
)
def test_get_features_nid_cov(cov_nid, test_files, churn, expected):
    assert get_features_nid_cov(cov_nid, test_files, churn) == expected