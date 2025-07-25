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

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

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

    agents.forEach(function(member) {
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

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);

    timestep = document.getElementById("timestep").value;
    document.getElementById("timestepValue").innerHTML = timestep;
    agents.forEach(function(member) {
        updateAgents(member, agents, timestep);
    });

    agents.forEach(function(member) {
        member.getData("agent").position.copy(member.position);
        member.getData('agent').material = agentMat;
    });

    renderer.render(scene, camera);
};

animate();