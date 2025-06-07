import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let agents = [];
let walls = [];
const COUNT = 150;
const RADIUS = 1;
const MAXSPEED = 10;
const HORIZON = 50;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function getPostition() {
    return [Math.random() * 90 - 45, Math.random() * 40 - 45];
}

function getVelocity() {
    let theta = Math.random() * Math.PI * 2;
    let speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

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

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();
        let pos = getPostition();  

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push(new Agent(
            pos[0], 2, pos[1],
            v[0], 0, v[1], 
            0, 0, 0,
            0, 2, 0,
            RADIUS, maxSpeed, maxForce, HORIZON, false, 3.0, k                   
        ));

        agents[i].setData("id", i);
    }

    let agentGeometry, agentMaterial, agent;

    agents.forEach(function(member) {
        agentGeometry = new THREE.CylinderGeometry(member.radius, 1, 4, 16);
        agentMaterial = new THREE.MeshLambertMaterial({
            color: 0x00ff00
        });
        agent = new THREE.Mesh(agentGeometry, agentMaterial);
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

        if (member.position.z >= 25) {
            member.target.z = 200;
        }
    });

    agents.forEach(function(member) {
        PHYSICS.update(member, agents);
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