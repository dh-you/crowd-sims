import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let agents = [];
let walls = [];
const COUNT = 75;
const RADIUS = 1;
const MAXSPEED = 7.5;
const HORIZON = 50;
const LENGTH = 100;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function getPostition() {
    return [Math.random() * 20 - 50, Math.random() * 100 - 50];
}

function getVelocity() {
    const theta = Math.random() * Math.PI * 2;
    const speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

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

    for (let i = 0; i < COUNT; i++) {
        const v = getVelocity();
        const pos = getPostition();

        const k = 1.5 + Math.random() * 1.5;
        const maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        const maxForce = 30 + Math.random() * 40;

        agents.push(new Agent(
            i,
            pos[0], 2, pos[1],            
            v[0], 0, v[1],
            0, 0, 0,
            0, 2, pos[1],            
            RADIUS, maxSpeed, maxForce, HORIZON, k
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
        if (member.position.z > LENGTH / 2 - RADIUS) {
            member.position.z = LENGTH / 2 - RADIUS;
        } else if (member.position.z < -LENGTH / 2 + RADIUS) {
            member.position.z = -LENGTH / 2 + RADIUS;
        }

        if (member.position.z < 0) {
            member.target.z = -12.5;
        } else {
            member.target.z = 12.5;
        }

        member.target.x = -32.5;
        member.maxSpeed = MAXSPEED;

        if (member.position.x >= -30 && member.position.x <= 30) {
            if (member.position.z < 0 && member.position.z > -15 && member.position.z < -10) {
                member.maxSpeed = MAXSPEED * 1.5;
            } else if (member.position.z > 10 && member.position.z < 15) {
                    member.maxSpeed = MAXSPEED * 1.5;
            }
        }

        if (Math.abs(member.position.x + 32.5) <= 7.5 && member.position.z > -15 && member.position.z < -10) {
            member.target.x = 200;
        } else if (Math.abs(member.position.x + 32.5) <= 7.5 && member.position.z > 10 && member.position.z < 15) {
            member.target.x = 200;
        }

        if (member.position.x >= -30 && member.position.z > -15 && member.position.z < -10) {
            member.target.x = 200;
        } else if (member.position.x >= -30 && member.position.z > 10 && member.position.z < 15) {
            member.target.x = 200;
        }

        if (member.position.x >= 25) {
            member.target.x = 200;
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