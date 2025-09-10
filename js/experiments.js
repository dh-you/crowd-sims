import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let timestep;

const CONFIG = {
    RADIUS: 1,
    MAXSPEED: 10,
    MAXFORCE: 30,
    HORIZON: 10,
    K: 2,
    AVOID: 5,
    SIDESTEP: 10,
}

let pauseButton = document.getElementById("pause");
pauseButton.addEventListener("click", pauseButtonLogic);

let downloadButton = document.getElementById("download");
downloadButton.addEventListener("click", downloadButtonLogic);

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera, world } = createScene();
init();

function pauseButtonLogic() {
    console.log("pause");
    world.pause = !world.pause;
    if (!world.pause) {
        pauseButton.src = "../resources/icons/8666604_pause_icon.png";
    } else {
        pauseButton.src = "../resources/icons/8666634_play_icon.png";
    }
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
    console.log("download");
    if (Object.keys(world.positions).length > 0) {
        let textFile = parsePositions();
        let a = document.createElement('a');
        a.href = "data:application/octet-stream," + encodeURIComponent(textFile);
        a.download = 'trajectories.txt';
        a.click();
    }
}

function init() {
    agents.push(new Agent(
        0,
        0, 2, -50,
        0, 0, 0,
        0, 0, 0,
        0, 2, 50,
        CONFIG.RADIUS, CONFIG.MAXSPEED + 5, CONFIG.MAXFORCE, CONFIG.HORIZON,
        CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP
    ));

    agents.push(new Agent(
        1,
        0, 2, -30,
        0, 0, 0,
        0, 0, 0,
        0, 2, 50,
        CONFIG.RADIUS, CONFIG.MAXSPEED, CONFIG.MAXFORCE, CONFIG.HORIZON,
        CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP
    ));

    agents.forEach(function (member) {
        const agentGeometry = new THREE.CylinderGeometry(member.radius, 1, 4, 16);
        const agentMaterial = new THREE.MeshLambertMaterial({
            color: 0x00ff00
        });
        const agent = new THREE.Mesh(agentGeometry, agentMaterial);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        member.setData("agent", agent);
    });
}

function animate() {
    requestAnimationFrame(animate);

    if (!world.pause) {
        timestep = document.getElementById("timestep").value;
        document.getElementById("timestepValue").innerHTML = timestep;
        agents.forEach(function (member) {
            updateAgents(member, agents, timestep);
        });
    }
    world.frame++;
    world.positions[world.frame] = {};

    agents.forEach(function (member, index) {
        world.positions[world.frame][index] = { "x": member.position.x, "z": member.position.z, "rotation": member.getData("agent").rotation.z };

        member.getData("agent").position.copy(member.position);
        member.getData('agent').material = agentMat;
    });

    renderer.render(scene, camera);
};

animate();