import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let walls = [];
let timestep;

const CONFIG = {
    COUNT: 102,
    RADIUS: 1,
    MAXSPEED: 5,
    MAXFORCE: 30,
    HORIZON: 5,
    K: 2,
    AVOID: 10,
    SIDESTEP: 5,
}

let leftRows = [];
let rightRows = [];
let orders = [];
let aisle = [];
let rowNum;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function shuffle(rowNum, left) {
    const row = left ? leftRows[rowNum] : rightRows[rowNum];
    for (const agent of row) agent.target.x += left ? -4 : 4;

    const nextAgent = row.pop();
    if (nextAgent) {
        aisle[rowNum] = nextAgent;
    }
}

function init() {
    const wallsData = [
        [90, 4, false, new THREE.Vector3(14, 2, -5)],
        [100, 4, false, new THREE.Vector3(-14, 2, 0)],
        [11, 4, true, new THREE.Vector3(8.5, 2, 40)],
        [11, 4, true, new THREE.Vector3(-8.5, 2, 40)]
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

            const maxSpeed = (Math.random() * 4) - 2 + CONFIG.MAXSPEED;
            const k = (Math.random() * 3) - 1.5 + CONFIG.K;
            const horizon = (Math.random() * 10) - 5 + CONFIG.HORIZON;

            agents.push(new Agent(
                --CONFIG.COUNT,
                j, 2, i,
                0, 0, 1, 
                0, 0, 0,
                j, 2, i,
                CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, horizon, 
                k, CONFIG.AVOID, CONFIG.SIDESTEP,                  
            ));
            row.push(agents[agents.length-1]);
        } 
        leftRows.push(row.splice(0, 3));
        rightRows.push(row.splice(0, 3));
        // random order for left row and right row exiting
        orders.push([true, true, true, false, false, false].sort(() => Math.random() - 0.5));
        aisle.push(null);
    }
    rowNum = aisle.length-1;

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

for (let i = 0; i < aisle.length; i++) {
    const left = orders[i].pop();
    shuffle(i, left);
}

function animate() {
    requestAnimationFrame(animate);

    if (rowNum >= 0) {
        let agentInAisle = aisle[rowNum];

        if (!agentInAisle && orders[rowNum].length > 0) {
            // choose random agent to let out into the aisle
            const left = orders[rowNum].pop();
            shuffle(rowNum, left);
        }

        agentInAisle = aisle[rowNum];
        if (agentInAisle) {
            const reachedAisle = Math.abs(agentInAisle.position.x) < 1.5;

            // do not let next agent move until agent in the aisle has left 
            if (!reachedAisle) {
                agentInAisle.target.x = 0;
            } else {
                agentInAisle.target.z = 45;
                agentInAisle.setData("state", "EXITING");
                aisle[rowNum] = null;
            }
        } else if (orders[rowNum].length === 0) {
            rowNum--; // row has finished exiting their seats
        }
    }
    
    agents.forEach(function(member) {
        // navigate out of plane
        if (member.getData("state") == "EXITING") {
            if (member.position.z >= 44) {
                member.target.x = 200;
                member.setData("state", "EXITED");
            } else if (member.position.z < 40) {
                if (member.position.x > 2.5 - CONFIG.RADIUS) {
                    member.position.x = 2.5 - CONFIG.RADIUS;
                } else if (member.position.x < -2.5 + CONFIG.RADIUS) {
                    member.position.x = -2.5 + CONFIG.RADIUS;
                }
            }
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