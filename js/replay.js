const canvas = document.querySelector("#simulation");
const context = canvas.getContext("2d");
const sceneSelect = document.querySelector("#scene");
const layoutControl = document.querySelector("#layout-control");
const layoutSelect = document.querySelector("#layout");
const countSelect = document.querySelector("#count");
const playButton = document.querySelector("#play");
const restartButton = document.querySelector("#restart");
const speedSelect = document.querySelector("#speed");
const timeline = document.querySelector("#timeline");
const status = document.querySelector("#status");
const details = document.querySelector("#details");

let replay = null;
let frameIndex = 0;
let playing = true;
let lastTimestamp = null;
let frameRemainder = 0;
let catalog = [];
let activeLoad = 0;

function layoutName(layout) {
  return layout.split("x").join("–");
}

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
    `${replay.scene}${replay.layout ? ` · ${layoutName(replay.layout)} seating` : ""} · ` +
    `${replay.agent_count} agents · ${(replay.frames[frameIndex] ?? []).length} remaining · ` +
    `frame ${frameIndex + 1}/${replay.frames.length}`;
}

function selectedEntry() {
  const count = Number(countSelect.value);
  return catalog.find(
    entry => entry.scene === sceneSelect.value && entry.count === count &&
      (entry.scene !== "airplane" || entry.layout === layoutSelect.value),
  );
}

function populateLayouts(scene, requestedLayout = null) {
  layoutControl.hidden = scene !== "airplane";
  const layouts = [...new Set(catalog
    .filter(entry => entry.scene === scene && entry.layout)
    .map(entry => entry.layout))]
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  layoutSelect.replaceChildren(...layouts.map(layout => {
    const option = document.createElement("option");
    option.value = layout;
    option.textContent = layoutName(layout) + (layout.split("x").length === 3 ? " (two aisles)" : "");
    return option;
  }));
  if (layouts.length) {
    layoutSelect.value = layouts.includes(requestedLayout) ? requestedLayout
      : layouts.includes("3x3") ? "3x3" : layouts[0];
  }
}

function populateCounts(scene, requestedCount = null) {
  const entries = catalog
    .filter(entry => entry.scene === scene &&
      (scene !== "airplane" || entry.layout === layoutSelect.value))
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
    : String(entries.at(-1)?.count ?? "");
}

async function loadReplay(entry) {
  const loadId = ++activeLoad;
  if (!entry) {
    replay = null;
    playing = false;
    playButton.disabled = restartButton.disabled = timeline.disabled = true;
    details.hidden = true;
    drawFrame();
    status.textContent = "No replay is available for this selection.";
    return;
  }

  replay = null;
  playing = false;
  frameIndex = 0;
  lastTimestamp = null;
  frameRemainder = 0;
  playButton.disabled = restartButton.disabled = timeline.disabled = true;
  details.hidden = true;
  status.textContent = `Loading ${entry.scene}${entry.layout ? ` ${layoutName(entry.layout)}` : ""} (${entry.count} agents)…`;
  drawFrame();

  const url = new URL(window.location);
  url.searchParams.set("scene", entry.scene);
  url.searchParams.set("count", entry.count);
  if (entry.layout) {
    url.searchParams.set("layout", entry.layout);
  } else {
    url.searchParams.delete("layout");
  }
  history.replaceState(null, "", url);

  try {
    const response = await fetch(`../data/replays/${entry.file}`, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const loaded = await response.json();
    if (loadId !== activeLoad) {
      return;
    }
    if (!Array.isArray(loaded.frames) || loaded.frames.length === 0) {
      throw new Error("the replay contains no frames");
    }
    if (loaded.complete === false) {
      throw new Error("the replay is incomplete");
    }
    if (loaded.scene !== entry.scene || loaded.agent_count !== entry.count ||
        (entry.scene === "airplane" && (loaded.layout ?? "3x3") !== entry.layout)) {
      throw new Error("the replay does not match the selection");
    }
    replay = { ...loaded, layout: entry.layout };

    canvas.width = replay.width;
    canvas.height = replay.height;
    timeline.max = replay.frames.length - 1;
    playing = true;
    playButton.textContent = "Pause";
    const preferredSpeeds = replay.preferred_speed_range;
    details.hidden = !Array.isArray(preferredSpeeds);
    details.textContent = Array.isArray(preferredSpeeds)
      ? `Preferred agent speeds: ${preferredSpeeds.join("–")}` : "";
    drawFrame();
  } catch (error) {
    if (loadId !== activeLoad) {
      return;
    }
    replay = null;
    playing = false;
    playButton.textContent = "Play";
    status.textContent =
      "Could not load this replay. Please try another selection.";
    console.error(error);
  } finally {
    if (loadId === activeLoad) {
      playButton.disabled = restartButton.disabled = timeline.disabled = !replay;
    }
  }
}

async function loadCatalog() {
  try {
    const response = await fetch("../data/replays/manifest.json", { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const manifest = await response.json();
    if (!Array.isArray(manifest.replays) || manifest.replays.length === 0) {
      throw new Error("the replay manifest is empty");
    }
    catalog = manifest.replays.map(entry => ({
      ...entry,
      layout: entry.scene === "airplane" ? entry.layout ?? "3x3" : null,
    }));

    const params = new URLSearchParams(window.location.search);
    const requestedScene = params.get("scene");
    const requestedCount = Number(params.get("count"));
    const requestedLayout = params.get("layout");
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
    populateLayouts(sceneSelect.value, requestedLayout);
    populateCounts(sceneSelect.value, requestedCount);
    await loadReplay(selectedEntry());
  } catch (error) {
    playing = false;
    playButton.textContent = "Play";
    status.textContent =
      "No replays are available yet.";
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
  populateLayouts(sceneSelect.value);
  populateCounts(sceneSelect.value);
  loadReplay(selectedEntry());
});

layoutSelect.addEventListener("change", () => {
  populateCounts(sceneSelect.value, Number(countSelect.value));
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
