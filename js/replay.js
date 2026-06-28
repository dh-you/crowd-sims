const canvas = document.querySelector("#simulation");
const context = canvas.getContext("2d");
const sceneSelect = document.querySelector("#scene");
const countSelect = document.querySelector("#count");
const playButton = document.querySelector("#play");
const restartButton = document.querySelector("#restart");
const speedSelect = document.querySelector("#speed");
const timeline = document.querySelector("#timeline");
const status = document.querySelector("#status");

let replay = null;
let frameIndex = 0;
let playing = true;
let lastTimestamp = null;
let frameRemainder = 0;
let catalog = [];

function drawFrame() {
  context.fillStyle = "#1e1e1e";
  context.fillRect(0, 0, canvas.width, canvas.height);

  if (!replay) {
    return;
  }

  context.fillStyle = "#c8c8c8";
  for (const [x, y, width, height] of replay.walls) {
    context.fillRect(x, y, width, height);
  }

  context.fillStyle = "#ff6400";
  for (const [, x, y, radius] of replay.frames[frameIndex] ?? []) {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }

  timeline.value = frameIndex;
  status.textContent =
    `${replay.scene} · ${replay.agent_count} agents · ` +
    `frame ${frameIndex + 1}/${replay.frames.length}`;
}

function selectedEntry() {
  const count = Number(countSelect.value);
  return catalog.find(
    entry => entry.scene === sceneSelect.value && entry.count === count,
  );
}

function populateCounts(scene, requestedCount = null) {
  const entries = catalog
    .filter(entry => entry.scene === scene)
    .sort((a, b) => a.count - b.count);

  countSelect.replaceChildren(
    ...entries.map(entry => {
      const option = document.createElement("option");
      option.value = entry.count;
      option.textContent = entry.count;
      return option;
    }),
  );

  const requestedExists = entries.some(entry => entry.count === requestedCount);
  countSelect.value = requestedExists
    ? String(requestedCount)
    : String(entries.at(-1).count);
}

async function loadReplay(entry) {
  if (!entry) {
    return;
  }

  replay = null;
  frameIndex = 0;
  lastTimestamp = null;
  frameRemainder = 0;
  status.textContent = `Loading ${entry.scene} (${entry.count} agents)…`;
  drawFrame();

  const url = new URL(window.location);
  url.searchParams.set("scene", entry.scene);
  url.searchParams.set("count", entry.count);
  history.replaceState(null, "", url);

  try {
    const response = await fetch(`../data/replays/${entry.file}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    replay = await response.json();
    if (!Array.isArray(replay.frames) || replay.frames.length === 0) {
      throw new Error("the replay contains no frames");
    }

    canvas.width = replay.width;
    canvas.height = replay.height;
    timeline.max = replay.frames.length - 1;
    playing = true;
    playButton.textContent = "Pause";
    drawFrame();
  } catch (error) {
    replay = null;
    playing = false;
    playButton.textContent = "Play";
    status.textContent =
      `Could not load ${entry.scene} with ${entry.count} agents.`;
    console.error(error);
  }
}

async function loadCatalog() {
  try {
    const response = await fetch("../data/replays/manifest.json");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const manifest = await response.json();
    catalog = manifest.replays;
    if (!Array.isArray(catalog) || catalog.length === 0) {
      throw new Error("the replay manifest is empty");
    }

    const params = new URLSearchParams(window.location.search);
    const requestedScene = params.get("scene");
    const requestedCount = Number(params.get("count"));
    const scenes = [...new Set(catalog.map(entry => entry.scene))].sort();

    sceneSelect.replaceChildren(
      ...scenes.map(scene => {
        const option = document.createElement("option");
        option.value = scene;
        option.textContent = scene[0].toUpperCase() + scene.slice(1);
        return option;
      }),
    );

    sceneSelect.value = scenes.includes(requestedScene)
      ? requestedScene
      : scenes[0];
    populateCounts(sceneSelect.value, requestedCount);
    await loadReplay(selectedEntry());
  } catch (error) {
    playing = false;
    playButton.textContent = "Play";
    status.textContent =
      "No replay catalog found. Export a replay with --export-json first.";
    console.error(error);
  }
}

function animate(timestamp) {
  if (playing && replay) {
    if (lastTimestamp !== null) {
      const elapsedSeconds = (timestamp - lastTimestamp) / 1000;
      const speed = Number(speedSelect.value);
      frameRemainder += elapsedSeconds * replay.fps * speed;
      const framesToAdvance = Math.floor(frameRemainder);

      if (framesToAdvance > 0) {
        frameRemainder -= framesToAdvance;
        frameIndex = Math.min(frameIndex + framesToAdvance, replay.frames.length - 1);
        if (frameIndex === replay.frames.length - 1) {
          playing = false;
          playButton.textContent = "Play";
        }
        drawFrame();
      }
    }
    lastTimestamp = timestamp;
  } else {
    lastTimestamp = null;
  }

  requestAnimationFrame(animate);
}

sceneSelect.addEventListener("change", () => {
  populateCounts(sceneSelect.value);
  loadReplay(selectedEntry());
});

countSelect.addEventListener("change", () => loadReplay(selectedEntry()));

playButton.addEventListener("click", () => {
  if (!replay) {
    return;
  }
  if (frameIndex === replay.frames.length - 1) {
    frameIndex = 0;
  }
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  drawFrame();
});

restartButton.addEventListener("click", () => {
  if (!replay) {
    return;
  }
  frameIndex = 0;
  frameRemainder = 0;
  playing = true;
  playButton.textContent = "Pause";
  drawFrame();
});

timeline.addEventListener("input", () => {
  if (!replay) {
    return;
  }
  frameIndex = Number(timeline.value);
  frameRemainder = 0;
  drawFrame();
});

loadCatalog();
requestAnimationFrame(animate);
