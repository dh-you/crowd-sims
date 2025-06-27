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
const HORIZON = 25;
const EPSILON = 0.01;

let paintings = [];
let points = [];
const wAgent = 30;
const wPainting = 80;
const MINCOMFORT = 11;
const MAXCOMFORT = 21;

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
    return points
        .sort((p1, p2) => weightedScore(p2, agent, painting) - weightedScore(p1, agent, painting))
        .pop();
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
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, -40)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, -20)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, 0)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, 20)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + EPSILON, 7.5, 40)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, -40)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, -20)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, 0)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, 20)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - EPSILON, 7.5, 40)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-40, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-20, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(0, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(20, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(40, 7.5, -50 + EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-40, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-20, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(0, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(20, 7.5, 50 - EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(40, 7.5, 50 - EPSILON)],
    ]

    paintingsData.forEach(([width, height, vertical, position]) => {
        const painting = new Wall(width, height, vertical, position);
        painting.mesh.material = new THREE.MeshLambertMaterial({
            color: new THREE.Color(Math.random(), Math.random(), Math.random()),
        });
        scene.add(painting.mesh);
        paintings.push(painting);
    });

    const p = new FastPoissonDiskSampling({
        shape: [100, 100],
        radius: 2*RADIUS,
        tries: 20
    }); 

    points = p.fill();
    points = points.map(([x, z]) => new THREE.Vector3(x - 50, 2, z - 50));
    points = points.filter(p => {
        if (paintings.some(painting => painting.mesh.position.distanceTo(p) < MINCOMFORT)) return false;
        if (!paintings.some(painting => painting.mesh.position.distanceTo(p) < MAXCOMFORT)) return false;
        return true;
    });

    for (let i = 0; i < COUNT; i++) {
        const v = getVelocity();

        const k = 1.5 + Math.random() * 1.5;
        const maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        const maxForce = 10 + Math.random() * 20;

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
        agents[i].position = pos;
        agents[i].target = pos;
        agents[i].setData("viewingPosition", pos);
        agents[i].setData("viewing", true);
        agents[i].setData("time", 30 + Math.random() * 180);
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

    if (timer >= 60) {
        timer = 0;

        const viewingAgents = agents.filter(a => a.getData("viewing"));
        for (let i = viewingAgents.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [viewingAgents[i], viewingAgents[j]] = [viewingAgents[j], viewingAgents[i]];
        }

        for (let i = 0; i < 10; i++) {
            const agent = viewingAgents[i];
            
            points.push(agent.getData("viewingPosition"));
            agent.target = new THREE.Vector3(0, 2, 0);
            agent.setData("goingCenter", true);
            agent.setData("viewing", false);
        }
    }

    agents.forEach(agent => {
        if (agent.getData("goingCenter")) {
            const dist = agent.position.distanceTo(new THREE.Vector3(0, 2, 0));
            if (dist < MAXCOMFORT) {
                const painting = paintings[Math.floor(Math.random() * paintings.length)];
                const pos = generateViewingPosition(agent, painting.mesh);

                agent.target = pos;
                agent.setData("viewingPosition", pos);
                agent.setData("viewing", false);
                agent.setData("goingCenter", false);
            }
        }
    });

    agents.forEach(function(member) {
        if (!member.getData("viewing") && !member.getData("goingCenter")) {
            const dist = member.position.distanceTo(member.getData("viewingPosition"));
            if (dist < RADIUS) {
                member.setData("viewing", true);
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
        member.getData('agent').material = agentMat;
    });

    renderer.render(scene, camera);
};


animate();