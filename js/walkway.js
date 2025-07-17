import { LENGTH, createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let walls = [];
let timestep;

const CONFIG = {
    COUNT: 100,
    RADIUS: 1,
    MAXSPEED: 5,
    MAXFORCE: 20,
    HORIZON: 30,
}

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();

function init() {
    const wallData = [
        [60, 2, true, new THREE.Vector3(0, 1, 15)],
        [60, 2, true, new THREE.Vector3(0, 1, 10)],
        [60, 2, true, new THREE.Vector3(0, 1, -10)],
        [60, 2, true, new THREE.Vector3(0, 1, -15)],
    ];

    wallData.forEach(([width, height, vertical, position]) => {
        const wall = new Wall(width, height, vertical, position);
        scene.add(wall.mesh);
        walls.push(wall);
    });

    const streetMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });
    const streetGeometry = new THREE.PlaneGeometry(60, 5);

    const street1 = new THREE.Mesh(streetGeometry, streetMaterial);
    street1.castShadow = true;
    street1.receiveShadow = true;
    street1.rotation.set(Math.PI / 2, 0, 0);
    street1.position.set(0, 0.05, 12.5);
    scene.add(street1);

    const street2 = new THREE.Mesh(streetGeometry, streetMaterial);
    street2.castShadow = true;
    street2.receiveShadow = true;
    street2.rotation.set(Math.PI / 2, 0, 0);
    street2.position.set(0, 0.05, -12.5);
    scene.add(street2);

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const v = UTILS.getVelocity(CONFIG.MAXSPEED);
        const pos = UTILS.getPosition(-50, -30, -50, 50);

        const k = 1.5 + Math.random() * 1.5;
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i,
            pos[0], 2, pos[1],            
            v[0], 0, v[1],
            0, 0, 0,
            0, 2, pos[1],            
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

function animate() {
    requestAnimationFrame(animate);

    agents.forEach(function(member) {
        // wrap around
        if (member.position.z > LENGTH / 2 - CONFIG.RADIUS) {
            member.position.z = LENGTH / 2 - CONFIG.RADIUS;
        } else if (member.position.z < -LENGTH / 2 + CONFIG.RADIUS) {
            member.position.z = -LENGTH / 2 + CONFIG.RADIUS;
        }

        // choose left or right walkway
        member.target.z = member.position.z < 0 ? -12.5 : 12.5;
        member.target.x = -32.5;

        // boost speed if in walkway
        if (member.position.x >= -30 && member.position.x <= 30) {
            if (member.position.z < 0 && member.position.z > -15 && member.position.z < -10) {
                member.maxSpeed = CONFIG.MAXSPEED * 1.5;
            } else if (member.position.z > 10 && member.position.z < 15) {
                member.maxSpeed = CONFIG.MAXSPEED * 1.5;
            }
        }

        // funnel agents through walkway
        const nearEntry = Math.abs(member.position.x + 32.5) <= 7.5;
        const left = member.position.z > 10 && member.position.z < 15;
        const right = member.position.z > -15 && member.position.z < -10;
        if ((nearEntry && (left || right)) || (member.position.x >= -30 && (left || right)) ||  member.position.x >= 25) {
            member.target.x = 200;
        }
    });

    timestep = document.getElementById("timestep").value;
    document.getElementById("timestepValue").innerHTML = timestep;
    agents.forEach(function(member) {
        updateAgents(member, agents, timestep);
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