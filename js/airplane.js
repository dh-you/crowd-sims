import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let walls = [];

const CONFIG = {
    COUNT: 102,
    RADIUS: 1,
    MAXSPEED: 10,
    MAXFORCE: 50,
    HORIZON: 50
}

let rows = [];
let aisle = [];

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function init() {
    const wallsData = [
        [90, 4, false, new THREE.Vector3(14, 2, -5)],
        [100, 4, false, new THREE.Vector3(-14, 2, 0)],
    ];

    wallsData.forEach(([width, height, vertical, position]) => {
        const wall = new Wall(width, height, vertical, position);
        scene.add(wall.mesh);
        walls.push(wall);
    });

    const floorMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });

    const geometry1 = new THREE.PlaneGeometry(100, 28);
    const plane1 = new THREE.Mesh(geometry1, floorMaterial);
    plane1.castShadow = true;
    plane1.receiveShadow = true;    
    plane1.rotation.set(Math.PI / 2, 0, Math.PI / 2);
    plane1.position.set(0, UTILS.EPSILON, 0);
    scene.add(plane1);

    const seatMaterial = new THREE.MeshPhongMaterial({ color: 0x808080, side: THREE.DoubleSide });

    const geometry2 = new THREE.PlaneGeometry(3, 3);
    for (let i = -47; i < 37; i += 5) {
        let row = [];
        for (const j of [12, 8, 4, -12, -8, -4]) {
            const seat = new THREE.Mesh(geometry2, seatMaterial);
            seat.castShadow = true;
            seat.receiveShadow = true;
            seat.rotation.set(Math.PI / 2, 0, Math.PI / 2);
            seat.position.set(j, UTILS.EPSILON * 2, i);
            scene.add(seat);

            const v = UTILS.getVelocity(CONFIG.MAXSPEED);

            const k = 1.5 + Math.random() * 1.5;
            const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

            agents.push(new Agent(
                i,
                j, 2, i,
                v[0], 0, v[1], 
                0, 0, 0,
                j, 2, i,
                CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON, k                   
            ));
            row.push([j, i,]);
        }
        rows.push(row);
        aisle.push([0, i, false]);
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
        updateAgents(member, agents);
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