import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let gatherPoints = [];
let timestep;

const CONFIG = {
    COUNT: 100,
    RADIUS: 1,
    MAXSPEED: 5,
    MAXFORCE: 30,
    HORIZON: 5,
    K: 2,
    AVOID: 10,
    SIDESTEP: 5,
    SPAWN_JITTER: 5,
    ARRIVAL_THRESHOLD: 3,
}

let pauseButton = document.getElementById("pause");
pauseButton.addEventListener("click", pauseButtonLogic);

let downloadButton = document.getElementById("download");
downloadButton.addEventListener("click", downloadButtonLogic);

const agentGeometry = new THREE.CylinderGeometry(CONFIG.RADIUS, CONFIG.RADIUS, 4, 8);
const agentMat = new THREE.MeshLambertMaterial({ color: 0x00ff00 });

const { renderer, scene, camera, world } = createScene("shibuya1.png");

init();

function pauseButtonLogic() {
    world.pause = !world.pause;
    pauseButton.src = world.pause
        ? "../resources/icons/8666634_play_icon.png"
        : "../resources/icons/8666604_pause_icon.png";
}

function parsePositions() {
    let result = "";
    for (let i = 1; i <= world.frame; i++) {
        let curData = world.positions[i];
        let keys = Object.keys(curData);
        keys.forEach((k, idx) => {
            let curAgent = curData[k];
            result += curAgent.x.toFixed(4) + "," + curAgent.z.toFixed(4);
            result += (idx < keys.length - 1) ? "," : "\n";
        });
    }
    return result;
}

function downloadButtonLogic() {
    if (Object.keys(world.positions).length > 0) {
        let textFile = parsePositions();
        let a = document.createElement('a');
        a.href = "data:application/octet-stream," + encodeURIComponent(textFile);
        a.download = 'trajectories.txt';
        a.click();
    }
}

function init() {
    gatherPoints.push(new THREE.Vector3(37, 0, -8)); // top-mid
    gatherPoints.push(new THREE.Vector3(30, 0, 41)); // top-right
    gatherPoints.push(new THREE.Vector3(30, 0, -33)); // top-left
    gatherPoints.push(new THREE.Vector3(-19, 0, -40)); // bottom-left
    gatherPoints.push(new THREE.Vector3(-30, 0, 34)); // bottom-right

    // debug spheres
    const gpGeo = new THREE.SphereGeometry(0.5);
    const gpMat = new THREE.MeshLambertMaterial({ color: 0xff0000 });
    gatherPoints.forEach(gp => {
        const mesh = new THREE.Mesh(gpGeo, gpMat);
        mesh.position.copy(gp);
        scene.add(mesh);
    });

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const spawn = Math.floor(Math.random() * gatherPoints.length);
        let dest = Math.floor(Math.random() * gatherPoints.length);
        while (dest == spawn) dest = Math.floor(Math.random() * gatherPoints.length);

        const spawnPoint = gatherPoints[spawn];
        const destPoint = gatherPoints[dest];
        const jitter = new THREE.Vector3(
            (Math.random() - 0.5) * CONFIG.SPAWN_JITTER * 2,
            0,
            (Math.random() - 0.5) * CONFIG.SPAWN_JITTER * 2
        );
        
        const pos = spawnPoint.clone().add(jitter);
        const maxSpeed = Math.random() * 2 + (CONFIG.MAXSPEED - 2);

        agents.push(new Agent(
            i,
            pos.x, 2, pos.z,
            0, 0, 0,
            0, 0, 0,
            destPoint.x, 2, destPoint.z,
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON,
            CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP
        ));

        const agent = new THREE.Mesh(agentGeometry, agentMat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        agent.userData = {
            "id": i,
        };
        scene.add(agent);
        agents[i].setData("agent", agent);
    }
}

function animate() {
    requestAnimationFrame(animate);

    if (!world.pause) {
        timestep = document.getElementById("timestep").value;
        document.getElementById("timestepValue").innerHTML = timestep;

        agents.forEach(agent => {
            updateAgents(agent, agents, timestep);
        });
    }

    world.frame++;
    world.positions[world.frame] = {};

    agents.forEach((agent, index) => {
        const mesh = agent.getData("agent");
        world.positions[world.frame][index] = {
            x: agent.position.x,
            z: agent.position.z,
            rotation: mesh.rotation.z
        };
        mesh.position.copy(agent.position);
    });

    renderer.render(scene, camera);
}

animate();