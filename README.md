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

Airplane training uses a gated candidate workflow. It collects complete expert
episodes, trains the four-input velocity policy, checks seat-row motion across
multiple crowd sizes and random seeds, and promotes artifacts only when every
check passes:

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
exited, and registers completed recordings in `data/replays/manifest.json`.

For airplanes, the viewer has separate **Layout** and **Agents** selectors.
Supported seating configurations are 2–1, 2–2, 3–2, 3–3, 3–4, and the two-aisle
3–4–3 cabin. The airplane recordings listed in the catalog use the existing
`heterogeneous_closer` learned checkpoint, seed 42, and preferred speeds sampled
uniformly from 3–7. No retraining is required.

For example, open
<http://localhost:1337/pages/replay.html?scene=airplane&layout=3x4x3&count=170>.
Existing links such as `?scene=airplane&count=102` still select the 3–3 cabin.
The page plays exported trajectories; changing or pushing a checkpoint alone
does not update an existing recording.

Export or replace a specific learned-model replay with:

```powershell
python imitation\inference.py --scene airplane --layout 2x2 --candidate --artifact-tag heterogeneous_closer --preferred-speed-min 3 --preferred-speed-max 7 --frame-stride 5 --export-json
```

Omit `--count` to fill the selected cabin, or specify a smaller passenger count.
Airplane filenames include layout and count, such as `airplane-2x2-68.json`, so
different layouts with equal counts can coexist. Other scenes retain filenames
such as `queue-10.json`. `--frame-stride 5` keeps recordings smaller by storing
every fifth simulation step and adjusting playback FPS to preserve duration;
the model and physics still run at every simulation step.

To export every full cabin using the same checkpoint and speed range:

```powershell
"2x1", "2x2", "3x2", "3x3", "3x4", "3x4x3" | ForEach-Object {
    python imitation\inference.py --scene airplane --layout $_ --candidate --artifact-tag heterogeneous_closer --preferred-speed-min 3 --preferred-speed-max 7 --frame-stride 5 --export-json
}
```

Use `--max-frames` only as an optional safety limit. An export that hits the
limit with agents remaining is marked incomplete and is not added to the web
viewer. In `data/replays`, it is saved as `*.incomplete.json` so it cannot replace
a previously completed run. Provide a path after `--export-json` to override
the output location; explicit paths inside `data/replays` are also cataloged.
The retraining workflow now exports directly into this same web directory.

Run the Python checks with `python -m unittest discover -s imitation/tests`.
Browser checks are in `tests/test_replay_browser.py`; install Playwright and
Chromium (`python -m pip install playwright`, then
`python -m playwright install chromium`) and run
`python -m unittest discover -s tests`. An installed system Chromium is also
supported. The browser checks start their own local HTTP server.

## Repository layout

- `js/`, `pages/`, `resources/`: browser simulations and assets
- `py/`: standalone Python simulations
- `imitation/core/`: shared Python simulation primitives
- `imitation/scenes/`: behavioral-cloning scene adapters
- `imitation/collect_data.py`: expert trajectory collection
- `imitation/train_bc.py`: policy training
- `imitation/inference.py`: trained-policy playback
