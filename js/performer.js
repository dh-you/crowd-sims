import { createScene } from './environment.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let pickableObjects = [];
let selected = null;
let mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

const COUNT = 50;
const RADIUS = 1;
const MAXSPEED = 7.5;
const HORIZON = 100;

const MINCOMFORT = 10;
const MAXCOMFORT = 25;
const wAgent = 30;
const wPerformer = 80;

let performer = {
    x: 0,
    y: 2,
    z: 45,
};
let agents = [];
let points;

const pedestrianMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const onlookerMat = new THREE.MeshLambertMaterial({
    color: 0x0000ff
});
const performerMat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

const { renderer, scene, camera } = createScene();
init();
render();

function distance(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
}

function weightedScore(point, agent, performer) {
    return wAgent * distance(point, [agent.x, agent.z]) + wPerformer * distance(point, [performer.x, performer.z]);
}

function generateViewingPosition(agent) {
    return points
        .sort((p1, p2) => weightedScore(p2, agent, performer) - weightedScore(p1, agent, performer))
        .pop();
}

function getPostition() {
    return [Math.random() * 90 - 45, Math.random() * 30 - 15];
}

function getVelocity() {
    let theta = Math.random() * Math.PI * 2;
    let speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

function init() {
    // street 
    const streetMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });
    const streetGeometry = new THREE.PlaneGeometry(100, 30);
    const streetPlane = new THREE.Mesh(streetGeometry, streetMaterial);
    streetPlane.castShadow = true;
    streetPlane.receiveShadow = true;
    streetPlane.rotation.set(Math.PI / 2, 0, 0);
    streetPlane.position.set(0, 0.05, 0);
    scene.add(streetPlane);

    let agentGeometry, agent;
    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();
        let pos = getPostition();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push({
            id : i,
            x: pos[0],
            y: 2,
            z: pos[1],
            vx: v[0],
            vy: 0,
            vz: v[1],
            gx: 0,
            gy: 0,
            gz: 0,
            tx: 50 * (Math.random() < 0.5 ? -1 : 1),
            ty: 2,
            tz: pos[1],
            radius: RADIUS,
            maxSpeed: maxSpeed,
            maxForce: maxForce,
            horizon: HORIZON,
            yield: false,
            isWatching: false,
            k: k,
        })

        agentGeometry = new THREE.CylinderGeometry(RADIUS, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, pedestrianMat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        agent.userData = {
            "id": agents[i].id
        };
        scene.add(agent);
        agents[i].agent = agent;
        pickableObjects.push(agent);
    }

    // performer
    agent = new THREE.Mesh(agentGeometry, performerMat);
    agent.castShadow = true;
    agent.receiveShadow = true;
    agent.position.set(performer.x, performer.y, performer.z);
    scene.add(agent);

    // poisson disc points
    var p = new FastPoissonDiskSampling({
        shape: [100, 100],
        radius: 2 * RADIUS,
        tries: 20
    }); 

    points = p.fill();
    points = points.map(([x, z]) => [x - 50, z - 50]);
    points = points.filter(([x, z]) => distance([x, z], [performer.x, performer.z]) > MINCOMFORT && distance([x, z], [performer.x, performer.z]) < MAXCOMFORT);

    window.addEventListener("mousedown", mouseDown, false);
}

function mouseDown(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    selected = null;
    var intersects = raycaster.intersectObjects(pickableObjects, false);
    for (var i = 0; i < intersects.length; i++) {
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
        if (member.x < -50 + RADIUS) { 
            member.x = 50 - RADIUS;
            member.z *= -1;
        } else if (member.x > 50 - RADIUS) {
            member.x = -50 + RADIUS;
            member.z *= -1;
        }

        // prevent clipping into wall
        if (member.z > 50 - RADIUS) {
            member.z = 50 - RADIUS;
        }

        // pick onlookers
        if (selected != null && member.id == selected && !member.isWatching) {
            member.isWatching = true;
            member.yield = true;
            [member.tx, member.tz] = generateViewingPosition(member);
            console.log(points.length);
        }

        if (member.isWatching && member.z > 20) {
            member.yield = false;
            member.horizon = 1;
        }

        member.agent.position.x = member.x;
        member.agent.position.y = member.y;
        member.agent.position.z = member.z;
        member.agent.material = member.isWatching ? onlookerMat : pedestrianMat;

        PHYSICS.update(member, agents);
    });

    renderer.render(scene, camera);
};

animate();