import { createScene } from './environment.js';
import { Agent } from './agent.js';
import { Wall } from './wall.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let walls = [];
let timestep = 0.005;

const CONFIG = {
    COUNT: 150,
    RADIUS: 1,
    MAXSPEED: 10,
    MAXFORCE: 50,
    HORIZON: 1,
    K: 2,
    AVOID: 5,
    SIDESTEP: 5,    
    TRANSITIONCAP: 20,
    wAgent: 30,
    wPainting: 80,
    MINCOMFORT: 10,
    MIDCOMFORT: 20,
    MAXCOMFORT: 30,
    BLOCKED_THRESH: 1
}

let viewerCounts = {};
let paintings = [];
let viewingPoints = [];
let exitPoints = [];
let inTransition = 0;

let pickableObjects = [];
const raycaster = new THREE.Raycaster();

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();

// balance distance from agent and distance to painting preferences
function weightedScore(point, agent, painting) {
    return CONFIG.wAgent * point.distanceTo(agent.position) + CONFIG.wPainting * point.distanceTo(painting.position);
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

// choose least crowded painting
function choosePainting(agent) {
    let unblockedPaintings = [];

    paintings.forEach(function(painting) {
        if (painting.mesh.id != agent.getData("painting")) {
            const dir = painting.mesh.position.clone().sub(agent.position.clone()).normalize();
            raycaster.set(agent.position.clone(), dir);
            
            const intersects = raycaster.intersectObjects(pickableObjects, false);
            if (intersects.length < CONFIG.BLOCKED_THRESH) unblockedPaintings.push(painting.mesh);
        }
    });

    let minViewers = CONFIG.COUNT;
    let bestPainting = paintings[Math.floor(Math.random()*paintings.length)]; // in case all is blocked
    unblockedPaintings.forEach(function(painting) {
        if (viewerCounts[painting.id] < minViewers) {
            minViewers = viewerCounts[painting.id];
            bestPainting = painting;
        }
    });

    return bestPainting;
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
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + UTILS.EPSILON, 7.5, -33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + UTILS.EPSILON, 7.5, -11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + UTILS.EPSILON, 7.5, 11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(-50 + UTILS.EPSILON, 7.5, 33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - UTILS.EPSILON, 7.5, -33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - UTILS.EPSILON, 7.5, -11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - UTILS.EPSILON, 7.5, 11)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), false, new THREE.Vector3(50 - UTILS.EPSILON, 7.5, 33)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-33, 7.5, -50 + UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-11, 7.5, -50 + UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(11, 7.5, -50 + UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(33, 7.5, -50 + UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-33, 7.5, 50 - UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(-11, 7.5, 50 - UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(11, 7.5, 50 - UTILS.EPSILON)],
        [THREE.MathUtils.randFloat(min, max), THREE.MathUtils.randFloat(min, max), true, new THREE.Vector3(33, 7.5, 50 - UTILS.EPSILON)],
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

    // precompute onlooker positions
    const p = new FastPoissonDiskSampling({
        shape: [100, 100],
        radius: 2*CONFIG.RADIUS,
        tries: 20
    }); 

    let points = p.fill();

    viewingPoints = points.map(([x, z]) => new THREE.Vector3(x - 50, 2, z - 50));
    viewingPoints = viewingPoints.filter(p => {
        if (paintings.some(painting => painting.mesh.position.distanceTo(p) < CONFIG.MINCOMFORT)) return false;
        if (!paintings.some(painting => painting.mesh.position.distanceTo(p) < CONFIG.MIDCOMFORT)) return false;
        return true;
    });

    exitPoints = points.map(([x, z]) => new THREE.Vector3(x - 50, 2, z - 50));
    exitPoints = exitPoints.filter(p => {
        if (paintings.some(painting => painting.mesh.position.distanceTo(p) < CONFIG.MIDCOMFORT)) return false;
        if (!paintings.some(painting => painting.mesh.position.distanceTo(p) > CONFIG.MAXCOMFORT)) return false;
        return true;
    });

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i,
            0, 2, 0,
            0, 0, 0, 
            0, 0, 0,
            0, 2, 0,
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON,
            CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP                   
        ));

        const painting = paintings[Math.floor(Math.random() * paintings.length)];
        const pos = generateViewingPosition(
            agents[i],
            painting.mesh
        );
        agents[i].target = pos;
        agents[i].position = pos.clone().add(new THREE.Vector3(UTILS.EPSILON, 0, UTILS.EPSILON));
        agents[i].setData("viewingPosition", pos);
        agents[i].setData("painting", painting.mesh.id);
        agents[i].setData("state", "VIEWING");
        agents[i].setData("timer", Math.random() * 10);
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
        pickableObjects.push(agent);
    });
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
                // subtract time          
                member.setData("timer", member.getData("timer") - (delta * timestep / 0.005));
                
                // navigate agent to exit point if there aren't too many other exiting agents
                if (member.getData("timer") <= 0 && inTransition < CONFIG.TRANSITIONCAP) {
                    inTransition++;
                    member.target = generateExitPosition(member);
                    member.setData("state", "EXITING");
                    viewerCounts[member.getData("painting")]--;
                } else if (member.getData("timer") <= 0 && inTransition >= CONFIG.TRANSITIONCAP) {
                    member.setData("timer", Math.random() * 10);
                }
                break;
            
            case "EXITING":
                // choose crowd with small crowd for agent to navigate to after exiting crowd
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
                // set new timer for viewer once painting is reached
                if (member.position.distanceTo(member.getData("viewingPosition")) < 5) {
                    inTransition--;
                    member.setData("timer", Math.random() * 10);
                    member.setData("state", "VIEWING");
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
        member.getData("agent").material = agentMat;
    });

    renderer.render(scene, camera);
};

animate()