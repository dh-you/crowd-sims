# Crowd Simulations

Interactive crowd simulations with browser-based visualizations and Python
implementations. The `imitation` package adds a behavioral-cloning workflow for
collecting expert trajectories, training policies, and replaying learned
behavior.

## Setup

Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Browser simulations

Start the local server from the repository root:

```powershell
python server.py
```

Then open <http://localhost:1337>. Individual simulations are available under
`pages/`.

## Python simulations

Run a scene directly:

```powershell
python py\room.py
python py\airplane.py
python py\queue.py
```

Press Space to pause or resume a simulation.

## Behavioral cloning

Scenes used by the learning workflow live in `imitation/scenes/`. Each scene
provides a common interface for building the environment, extracting state and
action vectors, and determining when agents leave.

Run the three workflow stages from the repository root:

```powershell
python imitation\collect_data.py --scene queue
python imitation\train_bc.py --scene queue
python imitation\inference.py --scene queue
```

Replace `queue` with `airplane` to use the other configured scene.
Generated datasets, normalization statistics, and model weights are stored in
`imitation/<scene>/`.

Useful options:

```powershell
python imitation\collect_data.py --scene queue --episodes 8
python imitation\train_bc.py --scene queue --epochs 100 --hidden 64
python imitation\inference.py --scene queue --alpha 0.7 --count 10
```

Airplane training uses a gated candidate workflow. It collects complete,
phase-labelled expert episodes without changing the policy's five inputs,
trains three seeds, checks seat-row motion across multiple crowd sizes and
random seeds, and promotes the artifacts only when every check passes:

```powershell
python imitation\retrain_airplane.py
```

For a quicker smoke run, reduce the episode, epoch, and evaluation-seed counts:

```powershell
python imitation\retrain_airplane.py --episodes 2 --epochs 10 --evaluation-seeds 1
```

### Web replays

Export an inference run to the generic browser replay viewer:

```powershell
python imitation\inference.py --scene queue --count 10 --export-json
python server.py
```

Open <http://localhost:1337/pages/replay.html?scene=queue&count=10>. The
exporter runs without opening a Pygame window, continues until every agent has
exited, and writes `data/replays/<scene>-<count>.json`. Completed exports are
automatically added to the viewer's agent-count selector through
`data/replays/manifest.json`.

Use `--max-frames` only as an optional safety limit. An export that hits the
limit with agents remaining is marked incomplete and is not added to the web
viewer. Provide a path after `--export-json` to override the output location.

## Repository layout

- `js/`, `pages/`, `resources/`: browser simulations and assets
- `py/`: standalone Python simulations
- `imitation/core/`: shared Python simulation primitives
- `imitation/scenes/`: behavioral-cloning scene adapters
- `imitation/collect_data.py`: expert trajectory collection
- `imitation/train_bc.py`: policy training
- `imitation/inference.py`: trained-policy playback
