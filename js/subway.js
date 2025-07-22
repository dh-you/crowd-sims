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
    COUNT: 20,
    RADIUS: 1,
    MAXSPEED: 5,
    MAXFORCE: 30,
    HORIZON: 5,
    K: 2,
    AVOID: 1,
    SIDESTEP: 5,
}

const gaps = [-33, -11, 11, 33];  
let wallTargets = [];
let agentsTargets = [];
let assigned = [];
let floor;
let animationDone = false;

const group1Mat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const group2Mat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

const { renderer, scene, camera } = createScene();
init();

function nearestDoor(z) {
    return gaps.sort((a, b) => Math.abs(z - a) - Math.abs(z - b))[0];
}

function init() {
    const wallData = [
        [100, 4, false, new THREE.Vector3(50, 2, 0)],
        [15, 4, true, new THREE.Vector3(42.5, 2, -50)],
        [15, 4, true, new THREE.Vector3(42.5, 2, 50)],
        [12, 4, false, new THREE.Vector3(35, 2, -44)],
        [12, 4, false, new THREE.Vector3(35, 2, -22)],
        [12, 4, false, new THREE.Vector3(35, 2, 0)],
        [12, 4, false, new THREE.Vector3(35, 2, 22)],
        [12, 4, false, new THREE.Vector3(35, 2, 44)],
    ];

    wallData.forEach(([width, height, vertical, position]) => {
        position.z += 200;
        const wall = new Wall(width, height, vertical, position);
        scene.add(wall.mesh);
        walls.push(wall);
        wallTargets.push(new THREE.Vector3(wall.mesh.position.x, 2, wall.mesh.position.z - 200));
    });

    const floorMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });

    const geometry9 = new THREE.PlaneGeometry(100, 15);
    const plane9 = new THREE.Mesh(geometry9, floorMaterial);
    plane9.castShadow = true;
    plane9.receiveShadow = true;    
    plane9.rotation.set(Math.PI / 2, 0, Math.PI / 2);
    plane9.position.set(42.5, 0.05, 200);
    floor = plane9;
    scene.add(plane9);

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const pos = UTILS.getPosition(40, 47.5, -39, 39);
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i,
            pos[0], 2, pos[1],
            0, 0, 0,
            0, 0, 0,
            0, 2, 0,
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON, 
            CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP
        ));
        agents[i].setData("group", 1);
        agentsTargets.push(new THREE.Vector3(agents[i].position.x, 2, agents[i].position.z));
        agents[i].position.z += 200;
        assigned.push(false);
    }

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const pos = UTILS.getPosition(0, 23, -39, 39);
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i + CONFIG.COUNT,
            pos[0], 2, pos[1],
            0, 0, 0,
            0, 0, 0,
            0, 2, 0,
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON,
            CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP,
        ));
        agents[i + CONFIG.COUNT].setData("group", 2);
    }

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const agentGeometry1 = new THREE.CylinderGeometry(agents[i].radius, 1, 4, 16);
        const agent1 = new THREE.Mesh(agentGeometry1, group1Mat);
        agent1.castShadow = true;
        agent1.receiveShadow = true;
        scene.add(agent1);
        agents[i].setData("agent", agent1);

        const agentGeometry2 = new THREE.CylinderGeometry(agents[i + CONFIG.COUNT].radius, 1, 4, 16);
        const agent2 = new THREE.Mesh(agentGeometry2, group2Mat);
        agent2.castShadow = true;
        agent2.receiveShadow = true;
        scene.add(agent2);
        agents[i + CONFIG.COUNT].setData("agent", agent2);
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    // subway arriving at station animation
    if (!animationDone) {
        let done = true;

        // smoothly lerp walls 
        for (let i = 0; i < walls.length; i++) {
            walls[i].mesh.position.lerp(wallTargets[i], 0.005);

            if (walls[i].mesh.position.distanceTo(wallTargets[i]) > 0.1) {
                done = false;
            } else {
                walls[i].box.setFromObject(walls[i].mesh);
            }
        }

        // smoothly lerp floor
        floor.position.lerp(new THREE.Vector3(42.5, 0.05, 0), 0.005);
        if (floor.position.distanceTo(new THREE.Vector3(42.5, 0.05, 0)) > 0.1) {
            done = false;
        }

        // smoothly lerp agents
        for (let i = 0; i < agents.length; i++) {
            if (agents[i].getData("group") == 1) {
                agents[i].position.lerp(agentsTargets[i], 0.005);

                if (agents[i].position.distanceTo(agentsTargets[i]) > 0.1) {
                    done = false;
                }
            }
            agents[i].getData("agent").position.copy(agents[i].position);
            agents[i].getData("agent").material = agents[i].getData("group") == 1 ? group1Mat : group2Mat;
        }

        if (done) {
            animationDone = true;
        }

        renderer.render(scene, camera);
        return;
    }

    agents.forEach(function(member) {
        // navigate agents to subway doors and outside
        switch (member.getData("group")) {
            case 1:
                if (member.position.x > 35 + CONFIG.RADIUS && !assigned[member.id]) {
                    member.target.z = nearestDoor(member.position.z);
                    member.target.x = 35 - 0.5;
                } else if (!assigned[member.id]) {
                    [member.target.x, member.target.z] = UTILS.getPosition(0, 23, -39, 39);
                    assigned[member.id] = true;
                }
                break;
                
            case 2:
                if (member.position.x < 35 - CONFIG.RADIUS && !assigned[member.id]) {
                    member.target.z = nearestDoor(member.position.z);
                    member.target.x = 35 + 0.5;
                } else if (!assigned[member.id]) {
                    [member.target.x, member.target.z] = UTILS.getPosition(40, 47.5, -39, 39);
                    assigned[member.id] = true;
                } 
                break;
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
        member.getData("agent").material = member.getData("group") == 1 ? group1Mat : group2Mat;
    });

    renderer.render(scene, camera);
};

animate();