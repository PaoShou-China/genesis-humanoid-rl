"""Render paired physical rollouts side by side, clearly marking ended clips."""

import argparse
import json
from pathlib import Path


def main():
    """Require matching protocol/command hashes before rendering a visual comparison."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--labels", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((r / "manifest.json").read_text()) for r in args.runs]
    for key in ("protocol_sha256", "command_plan_sha256"):
        if (
            not all(key in manifest for manifest in manifests)
            or manifests[0][key] != manifests[1][key]
        ):
            parser.error("Videos require the same version-2 physical protocol and command plan")
    readers = [imageio.get_reader(str(run / "rollout.mp4")) for run in args.runs]
    writer = None
    try:
        fps = [reader.get_meta_data()["fps"] for reader in readers]
        if abs(fps[0] - fps[1]) > 1e-6:
            parser.error("Video frame rates differ")
        counts = [reader.count_frames() for reader in readers]
        if min(counts) < 1:
            parser.error("Empty video")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.output), fps=fps[0])
        font = ImageFont.load_default(size=18)
        last = [None, None]
        for frame in range(max(counts)):
            panels = []
            for index, reader in enumerate(readers):
                if frame < counts[index]:
                    last[index] = reader.get_data(frame)
                panel = Image.fromarray(last[index].copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 34), fill=(245, 247, 250))
                ended = " [clip ended; frame held]" if frame >= counts[index] else ""
                draw.text(
                    (10, 7),
                    f"{args.labels[index]} | {manifests[index].get('video_start_s', 0) + min(frame, counts[index] - 1) / fps[index]:.2f}s{ended}",
                    fill=(24, 35, 52),
                    font=font,
                )
                panels.append(np.asarray(panel))
            if panels[0].shape != panels[1].shape:
                parser.error("Video resolutions differ")
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
