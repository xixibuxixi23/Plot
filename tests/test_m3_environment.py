import pytest

from scripts.check_m3_environment import version_tuple


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.7.1+cu128", (2, 7, 1)),
        ("2.8.0a0+5228986c39.nv25.05", (2, 8, 0)),
        ("2.9.0.dev20260913", (2, 9, 0)),
    ],
)
def test_version_tuple_accepts_cuda_and_prerelease_versions(raw, expected):
    assert version_tuple(raw) == expected


def test_version_tuple_rejects_non_version_text():
    with pytest.raises(ValueError, match="cannot parse version"):
        version_tuple("unknown")
