import { createScene } from './environment.js';
import { Wall } from './wall.js';
import { Agent } from './agent.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let agents = [];
let walls = [];
const gaps = [-33, -11, 11, 33];  
let wallTargets = [];
let agentsTargets = [];
let assigned = [];
let floor;
const COUNT = 20;
const RADIUS = 1;
const MAXSPEED = 5;
const HORIZON = 100;

const group1Mat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const group2Mat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

let animationDone = false;

const { renderer, scene, camera } = createScene();
init();
render();

function getPostition(min, max) {
    return Math.random() * (max - min) + min;
}

function getVelocity() {
    let theta = Math.random() * Math.PI * 2;
    let speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

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

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push(new Agent(
            i,
            getPostition(40, 47.5), 2, getPostition(-39, 39),
            v[0], 0, v[1],
            0, 0, 0,
            0, 2, 0,
            RADIUS, maxSpeed, maxForce, HORIZON, k
        ));
        agents[i].setData("group", 1);
        agentsTargets.push(new THREE.Vector3(agents[i].position.x, 2, agents[i].position.z));
        agents[i].position.z += 200;
        assigned.push(false);
    }

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push(new Agent(
            i + COUNT,
            getPostition(0, 23), 2, getPostition(-39, 39),
            v[0], 0, v[1],
            0, 0, 0,
            0, 2, 0,
            RADIUS, maxSpeed, maxForce, HORIZON, k
        ));
        agents[i + COUNT].setData("group", 2);
    }

    let agentGeometry, agent;

    for (let i = 0; i < COUNT; i++) {
        agentGeometry = new THREE.CylinderGeometry(agents[i].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group1Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        agents[i].setData("agent", agent);

        agentGeometry = new THREE.CylinderGeometry(agents[i + COUNT].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group2Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        agents[i + COUNT].setData("agent", agent);
    }
}

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);
    
    if (!animationDone) {
        let done = true;

        for (let i = 0; i < walls.length; i++) {
            walls[i].mesh.position.lerp(wallTargets[i], 0.005);

            if (walls[i].mesh.position.distanceTo(wallTargets[i]) > 0.1) {
                done = false;
            } else {
                walls[i].box.setFromObject(walls[i].mesh);
            }
        }

        floor.position.lerp(new THREE.Vector3(42.5, 0.05, 0), 0.005);

        if (floor.position.distanceTo(new THREE.Vector3(42.5, 0.05, 0)) > 0.1) {
            done = false;
        }

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
        if (member.getData("group") == 1) {
            if (member.position.x > 35 + RADIUS && !assigned[member.id]) {
                member.target.z = nearestDoor(member.position.z);
                member.target.x = 35 - 0.5;
            } else if (!assigned[member.id]) {
                member.target.x = getPostition(0, 23);
                member.target.z = getPostition(-39, 39);
                assigned[member.id] = true;
            }
        } else {
           if (member.position.x < 35 - RADIUS && !assigned[member.id]) {
                member.target.z = nearestDoor(member.position.z);
                member.target.x = 35 + 0.5;
            } else if (!assigned[member.id]) {
                member.target.x = getPostition(40, 47.5);
                member.target.z = getPostition(-39, 39);
                assigned[member.id] = true;
            } 
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
        member.getData("agent").material = member.getData("group") == 1 ? group1Mat : group2Mat;
    });

    agents.forEach(function(member) {
        member.getData("agent").position.copy(member.position);
        member.getData("agent").material = member.getData("group") == 1 ? group1Mat : group2Mat;
    });

    renderer.render(scene, camera);
};

animate();