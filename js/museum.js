import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let agents = [];
let walls = [];
const COUNT = 150;
const RADIUS = 1;
const MAXSPEED = 2.5;
const HORIZON = 1;
const EPSILON = 0.01;

let viewerCounts = {};
let paintings = [];
let viewingPoints = [];
let exitPoints = [];
let inTransition = 0;
const TRANSITIONCAP = 10;
const wAgent = 30;
const wPainting = 80;
const MINCOMFORT = 11;
const MIDCOMFORT = 21;
const MAXCOMFORT = 31;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function weightedScore(point, agent, painting) {
    return wAgent * point.distanceTo(agent.position) + wPainting * point.distanceTo(painting.position);
}

function generateViewingPosition(agent, painting) {
    return viewingPoints
        .sort((p1, p2) => weightedScore(p2, agent, painting) - weightedScore(p1, agent, painting))
        .pop();
}

function generateExitPosition(agent) {
    return exitPoints
        .sort((p1, p2) => p1.distanceTo(agent.position) - p2.distanceTo(agent.position))
        .at(0);
}

function choosePainting(agent) {
    let minViewers = COUNT;
    let bestPainting;

    paintings.forEach(function(painting) {
        if (painting.mesh.id != agent.getData("painting")) {
            if (viewerCounts[painting.mesh.id] < minViewers) {
                minViewers = viewerCounts[painting.mesh.id];
                bestPainting = painting.mesh;
            }
        }
    });

    return bestPainting;
}

function getVelocity() {
    const theta = Math.random() * Math.PI * 2;
    const speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

function init() {
    const wallsData = [
        [100, 25, true, new THREE.Vector3(0, 2, -50)],
        [100, 25, false, new THREE.Vector3(50, 2, 0)],
        [100, 25, false, new THREE.Vector3(-50, 2, 0)],
        [100, 25, true, new THREE.Vector3(0, 2, 50)],
    ];

    wallsData.forEach(([width, height, vertical, position]) => {
        const wall = new Wall(width, height, vertical, position);
        scene.add(wall.mesh);
        walls.push(wall);
    });

    let min = 5;
    let max = 10;
    const paintingsData = [
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, -33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, -11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, 11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, 33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, -33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, -11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, 11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, 33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-33, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-11, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(11, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(33, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-33, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-11, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(11, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(33, 7.5, 50 - EPSILON)],
    ]

    paintingsData.forEach(([width, height, vertical, position]) => {
        const painting = new Wall(width, height, vertical, position);
        painting.mesh.material = new THREE.MeshLambertMaterial({
            color: new THREE.Color(Math.random(), Math.random(), Math.random()),
        });
        scene.add(painting.mesh);
        paintings.push(painting);
        viewerCounts[painting.mesh.id] = 0;
    });

    const p = new FastPoissonDiskSampling({
        shape: [100, 100],
        radius: 2*RADIUS,
        tries: 20
    }); 

    let points = p.fill();

    viewingPoints = points.map(([x, z]) => new THREE.Vector3(x - 50, 2, z - 50));
    viewingPoints = viewingPoints.filter(p => {
        if (paintings.some(painting => painting.mesh.position.distanceTo(p) < MINCOMFORT)) return false;
        if (!paintings.some(painting => painting.mesh.position.distanceTo(p) < MIDCOMFORT)) return false;
        return true;
    });

    exitPoints = points.map(([x, z]) => new THREE.Vector3(x - 50, 2, z - 50));
    exitPoints = exitPoints.filter(p => {
        if (paintings.some(painting => painting.mesh.position.distanceTo(p) < MIDCOMFORT)) return false;
        if (!paintings.some(painting => painting.mesh.position.distanceTo(p) > MAXCOMFORT)) return false;
        return true;
    });

    for (let i = 0; i < COUNT; i++) {
        const v = getVelocity();

        const k = 1.5 + Math.random() * 1.5;
        const maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        const maxForce = 30 + Math.random() * 40;

        agents.push(new Agent(
            i,
            0, 2, 0,
            v[0], 0, v[1], 
            0, 0, 0,
            0, 2, 0,
            RADIUS, maxSpeed, maxForce, HORIZON, k                   
        ));

        const painting = paintings[Math.floor(Math.random() * paintings.length)];
        const pos = generateViewingPosition(
            agents[i],
            painting.mesh
        );
        agents[i].target = pos;
        agents[i].position = pos.clone().add(new THREE.Vector3(EPSILON, 0, EPSILON));
        agents[i].setData("viewingPosition", pos);
        agents[i].setData("painting", painting.mesh.id);
        agents[i].setData("state", "VIEWING");
        agents[i].setData("timer", 30 + Math.random() * 300);
        viewerCounts[painting.mesh.id]++;
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

let prev = 0;
let timer = 0;

function animate(timestamp = 0) {
    requestAnimationFrame(animate);

    const delta = (timestamp - prev) / 1000;
    prev = timestamp;

    timer += delta;

    agents.forEach(function(member) {
        switch (member.getData("state")) {
            case "VIEWING":
                member.horizon = 1;
                member.setData("timer", member.getData("timer") - delta);
                member.target = member.getData("viewingPosition");
                if (member.getData("timer") <= 0 && inTransition < TRANSITIONCAP) {
                    inTransition++;
                    member.target = generateExitPosition(member);
                    member.setData("state", "EXITING");
                    viewerCounts[member.getData("painting")]--;
                } else if (member.getData("timer") <= 0 && inTransition >= TRANSITIONCAP) {
                    console.log(TRANSITIONCAP);
                    member.setData("timer", 10 + Math.random() * 100);
                }
                break;
            
            case "EXITING":
                member.horizon = 15;
                if (member.position.distanceTo(member.target) < 5) {
                    const painting = choosePainting(member);
                    viewingPoints.push(member.getData("viewingPosition"));
                    member.setData("viewingPosition", generateViewingPosition(member, painting));
                    member.target = member.getData("viewingPosition").clone();
                    member.setData("painting", painting.id);
                    member.setData("state", "WALKING");
                    viewerCounts[member.getData("painting")]++;
                }
                break;

            case "WALKING":
                member.horizon = 25;
                if (member.position.distanceTo(member.getData("viewingPosition")) < 5) {
                    inTransition--;
                    member.setData("timer", 30 + Math.random() * 300);
                    member.setData("state", "VIEWING");
                }
                break;
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


animate()