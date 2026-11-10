"""Write layout-aware recordings and the browser's replay catalog."""

import json
import os
from pathlib import Path
import tempfile


DEFAULT_AIRPLANE_LAYOUT = "3x3"


def capture_frame(agents):
    return [
        [agent.id, round(agent.position.x, 3), round(agent.position.y, 3),
         round(agent.radius, 3)]
        for agent in agents
    ]


def replay_key(entry):
    scene = entry["scene"]
    layout = (entry.get("layout") or DEFAULT_AIRPLANE_LAYOUT) if scene == "airplane" else ""
    return scene, layout, entry["count"]


def replay_filename(scene, count, layout=None):
    if scene == "airplane":
        return f"{scene}-{layout or DEFAULT_AIRPLANE_LAYOUT}-{count}.json"
    return f"{scene}-{count}.json"


def _write_json(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as output:
            temporary_path = Path(output.name)
            if compact:
                json.dump(value, output, separators=(",", ":"))
            else:
                json.dump(value, output, indent=2)
                output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_web_replay(replay, repo_root, output_path=None):
    """Register completed recordings saved in data/replays, including explicit paths.

    Legacy airplane catalog entries without a layout are treated as 3x3.
    An incomplete managed recording gets a diagnostic filename so it cannot
    overwrite a completed recording already advertised by the browser.
    """
    replay_dir = (Path(repo_root) / "data" / "replays").resolve()
    if output_path:
        destination = Path(output_path).resolve()
    else:
        destination = replay_dir / replay_filename(
            replay["scene"], replay["agent_count"], replay.get("layout"),
        )
    managed = destination.parent == replay_dir
    if managed and not replay["complete"]:
        destination = destination.with_name(destination.stem + ".incomplete.json")
    _write_json(destination, replay, compact=True)

    if managed and replay["complete"]:
        manifest_path = replay_dir / "manifest.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else {"replays": []})
        entry = {
            "scene": replay["scene"], "count": replay["agent_count"],
            "file": destination.name, "frames": len(replay["frames"]),
        }
        if replay["scene"] == "airplane":
            entry["layout"] = replay.get("layout") or DEFAULT_AIRPLANE_LAYOUT
        manifest["version"] = 2
        manifest["replays"] = [
            item for item in manifest["replays"] if replay_key(item) != replay_key(entry)
        ]
        manifest["replays"].append(entry)
        manifest["replays"].sort(key=replay_key)
        _write_json(manifest_path, manifest)
    return destination
