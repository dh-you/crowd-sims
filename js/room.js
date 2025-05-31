import { createScene } from './environment.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

const LENGTH = 100;

let agents = [];
const COUNT = 150;
const RADIUS = 1;
const MAXSPEED = 10;
const HORIZON = 50;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function getPostition() {
    return [Math.random() * 40 - 45, Math.random() * 90 - 45];
}

function getVelocity() {
    let theta = Math.random() * Math.PI * 2;
    let speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

function init() {
    // walls 
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x222222, side: THREE.DoubleSide});

    const geometry1 = new THREE.PlaneGeometry(100, 4);
    const plane1 = new THREE.Mesh(geometry1, wallMaterial);
    plane1.rotation.set(Math.PI, 0, 0);
    plane1.position.set(0, 2, -50);
    scene.add(plane1);

    const geometry2 = new THREE.PlaneGeometry(75, 4);
    const plane2 = new THREE.Mesh(geometry2, wallMaterial);
    plane2.rotation.set(0, Math.PI / 2, 0);
    plane2.position.set(50, 2, -12.5);
    scene.add(plane2);

    const geometry3 = new THREE.PlaneGeometry(75, 4);
    const plane3 = new THREE.Mesh(geometry3, wallMaterial);
    plane3.rotation.set(0, Math.PI / 2, 0);
    plane3.position.set(-50, 2, -12.5);
    scene.add(plane3);

    const geometry4 = new THREE.PlaneGeometry(40, 4);
    const plane4 = new THREE.Mesh(geometry4, wallMaterial);
    plane4.rotation.set(Math.PI, 0, 0);
    plane4.position.set(30, 2, 25);
    scene.add(plane4);

    const geometry5 = new THREE.PlaneGeometry(40, 4);
    const plane5 = new THREE.Mesh(geometry5, wallMaterial);
    plane5.rotation.set(Math.PI, 0, 0);
    plane5.position.set(-30, 2, 25);
    scene.add(plane5);

    const geometry6 = new THREE.PlaneGeometry(25, 4);
    const plane6 = new THREE.Mesh(geometry6, wallMaterial);
    plane6.rotation.set(0, Math.PI / 2, 0);
    plane6.position.set(10, 2, 37.5);
    scene.add(plane6);

    const geometry7 = new THREE.PlaneGeometry(25, 4);
    const plane7 = new THREE.Mesh(geometry7, wallMaterial);
    plane7.rotation.set(0, Math.PI / 2, 0);
    plane7.position.set(-10, 2, 37.5);
    scene.add(plane7);

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();
        let pos = getPostition();  

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push({
            id : i,
            x: pos[1],
            y: 2,
            z: pos[0],
            vx: v[0],
            vy: 0,
            vz: v[1],
            gx: 0,
            gy: 0,
            gz: 0,
            tx: 0,
            ty: 2,
            tz: 0,
            radius: RADIUS,
            horizon: HORIZON,
            k: k,
            maxSpeed: maxSpeed,
            maxForce: maxForce,
            yield: false,
        })
    }

    let agentGeometry, agentMaterial, agent;

    agents.forEach(function(member) {
        agentGeometry = new THREE.CylinderGeometry(member.radius, 1, 4, 16);
        agentMaterial = new THREE.MeshLambertMaterial({
            color: 0x00ff00
        });
        agent = new THREE.Mesh(agentGeometry, agentMaterial);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        member.agent = agent;
    });
}

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);

    agents.forEach(function(member) {
        member.agent.position.x = member.x;
        member.agent.position.y = member.y;
        member.agent.position.z = member.z;
        member.agent.material = agentMat;

        if (member.z < 25) {
            if (member.x > LENGTH / 2 - RADIUS) {
                member.x = LENGTH / 2 - RADIUS; 
            } else if (member.x < -LENGTH / 2 + RADIUS) {
                member.x = -LENGTH / 2 + RADIUS; 
            } else if (member.z > 25-RADIUS && (member.x < -10 || member.x > 10)) {
                member.z = 25-RADIUS;
            } else if (member.z < -LENGTH / 2 + RADIUS) {
                member.z = -LENGTH / 2 + RADIUS;
            }
            
            if (member.x > 10) {
                member.tx = -10;
            } else if (member.x < -10) {
                member.tx = 10;
            } else {
                member.tx = 0;
            }

            member.tz = 27.5;
        }

        if (member.z >= 25) {
            if (member.x > 10 - RADIUS) {
                member.x = 10 - RADIUS;
            } else if (member.x < -10 + RADIUS) {
                member.x = -10 + RADIUS;
            } else if (member.z < -LENGTH / 2 + RADIUS) {
                member.z = -LENGTH / 2 + RADIUS;
            } else if (member.x < -10 || member.x > 10) {
                member.z = -RADIUS;
            }

            member.tz = 200;
        }

        PHYSICS.update(member, agents);
    });
    renderer.render(scene, camera);
};

animate();