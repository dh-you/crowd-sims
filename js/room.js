import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let walls = [];

const CONFIG = {
    COUNT: 150,
    RADIUS: 1,
    MAXSPEED: 10,
    MAXFORCE: 50,
    HORIZON: 50
}

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function init() {
    const wallsData = [
        [100, 4, true, new THREE.Vector3(0, 2, -50)],
        [75, 4, false, new THREE.Vector3(50, 2, -12.5)],
        [75, 4, false, new THREE.Vector3(-50, 2, -12.5)],
        [40, 4, true, new THREE.Vector3(30, 2, 25)],
        [40, 4, true, new THREE.Vector3(-30, 2, 25)],
        [25, 4, false, new THREE.Vector3(10, 2, 37.5)],
        [25, 4, false, new THREE.Vector3(-10, 2, 37.5)],
    ];

    wallsData.forEach(([width, height, vertical, position]) => {
        const wall = new Wall(width, height, vertical, position);
        scene.add(wall.mesh);
        walls.push(wall);
    });

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const v = UTILS.getVelocity(CONFIG.MAXSPEED);
        const pos = UTILS.getPosition(-45, 45, -45, -5);

        const k = 1.5 + Math.random() * 1.5;
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i,
            pos[0], 2, pos[1],
            v[0], 0, v[1], 
            0, 0, 0,
            0, 2, 0,
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON, k                   
        ));
    }

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

    agents.forEach(function(member) {
        // navigate agents to entry
        if (member.position.z < 25) {            
            if (member.position.x > 10) {
                member.target.x = -10;
            } else if (member.position.x < -10) {
                member.target.x = 10;
            } else {
                member.target.x = 0;
            }
            member.target.z = 27.5;
        }

        // navigate agents to exit
        if (member.position.z >= 25) {
            member.target.z = 200;
        }
    });

    agents.forEach(function(member) {
        updateAgents(member, agents);
    });

    agents.forEach(function(agent) {
        walls.forEach(function(wall) {
            wall.collisionResolve(agent);
        });
    });

    agents.forEach(function(member) {
        member.getData("agent").position.copy(member.position);
        member.getData('agent').material = agentMat;
    });

    renderer.render(scene, camera);
};

animate();