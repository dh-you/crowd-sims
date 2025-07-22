import { LENGTH, createScene } from './environment.js';
import { Agent } from './agent.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let timestep;

const CONFIG = {
    COUNT: 50,
    RADIUS: 1,
    MAXSPEED: 7.5,
    HORIZON: 10,
    K: 2,
    AVOID: 5,
    SIDESTEP: 5,
    MINCOMFORT: 10,
    MAXCOMFORT: 25,
    MAXFORCE: 50,
    wAgent: 30,
    wPerformer: 80
}

let performer = new THREE.Vector3(0, 2, 45);
let points;

let pickableObjects = [];
let selected = null;
let mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

const pedestrianMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const onlookerMat = new THREE.MeshLambertMaterial({
    color: 0x0000ff
});
const performerMat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

const agentGeometry = new THREE.CylinderGeometry(CONFIG.RADIUS, 1, 4, 16);

const { renderer, scene, camera } = createScene();
init();
render();

// balance distance from agent and distance to performer preferences
function weightedScore(point, agent, performer) {
    return CONFIG.wAgent * point.distanceTo(agent.position) + CONFIG.wPerformer * point.distanceTo(performer);
}

function generateViewingPosition(agent) {
    return points
        .sort((p1, p2) => weightedScore(p2, agent, performer) - weightedScore(p1, agent, performer))
        .pop();
}

function init() {
    const streetMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });
    const streetGeometry = new THREE.PlaneGeometry(100, 30);
    const streetPlane = new THREE.Mesh(streetGeometry, streetMaterial);
    streetPlane.castShadow = true;
    streetPlane.receiveShadow = true;
    streetPlane.rotation.set(Math.PI / 2, 0, 0);
    streetPlane.position.set(0, 0.05, 0);
    scene.add(streetPlane);

    for (let i = 0; i < CONFIG.COUNT; i++) {
        const pos = UTILS.getPosition(-45, 45, -15, 15);
        const maxSpeed = Math.random() * (CONFIG.MAXSPEED - 5) + 5;

        agents.push(new Agent(
            i,
            pos[0], 2, pos[1],
            0, 0, 0,
            0, 0, 0,
            50 * (Math.random() < 0.5 ? -1 : 1), 2, pos[1],
            CONFIG.RADIUS, maxSpeed, CONFIG.MAXFORCE, CONFIG.HORIZON, 
            CONFIG.K, CONFIG.AVOID, CONFIG.SIDESTEP,
        ));
        agents[i].setData("state", "WALKING");

        const agent = new THREE.Mesh(agentGeometry, pedestrianMat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        agent.userData = {
            "id": i,
        };
        scene.add(agent);
        agents[i].setData("agent", agent);
        pickableObjects.push(agent);
    }

    const agent = new THREE.Mesh(agentGeometry, performerMat);
    agent.castShadow = true;
    agent.receiveShadow = true;
    agent.position.set(performer.x, performer.y, performer.z);
    scene.add(agent);

    // precompute onlooker positions
    const p = new FastPoissonDiskSampling({
        shape: [100, 100],
        radius: 2 * CONFIG.RADIUS,
        tries: 20
    }); 

    points = p.fill();
    points = points.map(([x, z]) => new THREE.Vector3(x - 50, 0, z - 50));
    points = points.filter(p => {
        const d = p.distanceTo(performer);
        return d > CONFIG.MINCOMFORT && d < CONFIG.MAXCOMFORT;
    });

    window.addEventListener("mousedown", mouseDown, false);
}

function mouseDown(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    selected = null;
    const intersects = raycaster.intersectObjects(pickableObjects, false);
    for (let i = 0; i < intersects.length; i++) {
        selected = intersects[i].object.userData.id;
    }
}

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);

    agents.forEach(function(member) {
        // wrap around
        if (member.position.x < -LENGTH / 2 + CONFIG.RADIUS) { 
            member.position.x = LENGTH / 2 - CONFIG.RADIUS;
            member.position.z *= -1;
        } else if (member.position.x > LENGTH / 2 - CONFIG.RADIUS) {
            member.position.x = -LENGTH / 2 + CONFIG.RADIUS;
            member.position.z *= -1;
        }

        // navigate agent towards viewing position
        if (selected != null && member.id == selected && member.getData("state") == "WALKING") {
            member.setData("state", "VIEWING");
            member.target = generateViewingPosition(member);
        }

        // lower agent horizon once in onlooker crowd
        if (member.getData("state") == "VIEWING" && member.position.z > 20) {
            member.horizon = 1;
        }
    });

    timestep = document.getElementById("timestep").value;
    document.getElementById("timestepValue").innerHTML = timestep;
    agents.forEach(function(member) {
        updateAgents(member, agents, timestep);
    });

    agents.forEach(function(member) {
        member.getData("agent").position.copy(member.position);
        member.getData("agent").material = member.getData("state") == "VIEWING" ? onlookerMat : pedestrianMat;
    });

    renderer.render(scene, camera);
};

animate();