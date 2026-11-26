"""Prefetch the pinned G1 asset set atomically and emit a local hash manifest."""

import argparse
import os
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    """Download before training; never vendor third-party meshes into this repository."""
    from humanoid.config import ROOT, read_yaml
    from humanoid.provenance import sha256, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    spec = read_yaml(ROOT / "configs/robots/g1_29dof.yaml")
    root = (
        Path(os.environ.get("GENESISLAB_ASSET_CACHE", Path.home() / ".cache/genesislab/assets"))
        / "unitree_g1"
    )
    base = (
        "https://raw.githubusercontent.com/mujocolab/mjlab/"
        + spec["asset_source_commit"]
        + "/src/mjlab/asset_zoo/robots/unitree_g1/xmls/"
    )

    def fetch(relative):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            temporary = target.with_name(target.name + ".part")
            try:
                with urllib.request.urlopen(base + relative, timeout=30) as response:
                    temporary.write_bytes(response.read())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return {"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target)}

    xml = fetch("g1.xml")
    if xml["sha256"] != spec["asset_xml_sha256"]:
        raise ValueError("Cached G1 XML differs from the audited model; no files were deleted")
    document = ET.parse(root / "g1.xml").getroot()
    meshes = ["assets/" + mesh.attrib["file"] for mesh in document.findall(".//asset/mesh")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifest = [xml, *pool.map(fetch, meshes)]
    write_json(args.output, {"asset_source_commit": spec["asset_source_commit"], "files": manifest})
    print(f"Verified pinned XML and recorded {len(manifest) - 1} mesh hashes")


if __name__ == "__main__":
    main()
