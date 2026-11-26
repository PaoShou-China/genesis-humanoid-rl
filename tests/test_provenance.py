"""Protect source snapshots from accidentally including weights and transient data."""

import zipfile

from humanoid.provenance import archive_source, sha256


def test_source_snapshot_is_self_contained_and_excludes_runs(tmp_path):
    digest = archive_source(tmp_path)
    snapshot = tmp_path / "source_snapshot.zip"
    assert digest == sha256(snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        names = archive.namelist()
        assert "scripts/train.py" in names
        assert "configs/robots/g1_29dof.yaml" in names
        assert not any(name.startswith(("outputs/", "logs/", ".git/")) for name in names)
