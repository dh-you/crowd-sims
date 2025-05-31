import { createScene } from './environment.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

let agents = [];
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

function init() {
    // walls 
    const wallMaterial = new THREE.MeshStandardMaterial({
        color: 0xa2a4a8,   
        metalness: 0.1,       
        roughness: 0.2,       
        side: THREE.DoubleSide
    });

    const floorMaterial = new THREE.MeshPhongMaterial({ color: 0x555555, side: THREE.DoubleSide });

    const geometry1 = new THREE.PlaneGeometry(100, 4);
    const plane1 = new THREE.Mesh(geometry1, wallMaterial);
    plane1.receiveShadow = true;    
    plane1.rotation.set(0, Math.PI / 2, 0);
    plane1.position.set(50, 2, 0);
    scene.add(plane1);

    const geometry2 = new THREE.PlaneGeometry(15, 4);
    const plane2 = new THREE.Mesh(geometry2, wallMaterial);
    plane2.receiveShadow = true;    
    plane2.rotation.set(Math.PI, 0, 0);
    plane2.position.set(42.5, 2, -50);
    scene.add(plane2);

    const geometry3 = new THREE.PlaneGeometry(15, 4);
    const plane3 = new THREE.Mesh(geometry3, wallMaterial);
    plane3.castShadow = true;
    plane3.receiveShadow = true;    
    plane3.rotation.set(Math.PI, 0, 0);
    plane3.position.set(42.5, 2, 50);
    scene.add(plane3);

    const geometry4 = new THREE.PlaneGeometry(12, 4);
    const plane4 = new THREE.Mesh(geometry4, wallMaterial);
    plane4.castShadow = true;
    plane4.receiveShadow = true;    
    plane4.rotation.set(0, Math.PI / 2, 0);
    plane4.position.set(35, 2, -44);
    scene.add(plane4);

    const geometry5 = new THREE.PlaneGeometry(12, 4);
    const plane5 = new THREE.Mesh(geometry5, wallMaterial);
    plane5.castShadow = true;
    plane5.receiveShadow = true;    
    plane5.rotation.set(0, Math.PI / 2, 0);
    plane5.position.set(35, 2, -22);
    scene.add(plane5);

    const geometry6 = new THREE.PlaneGeometry(12, 4);
    const plane6 = new THREE.Mesh(geometry6, wallMaterial);
    plane6.castShadow = true;
    plane6.receiveShadow = true;    
    plane6.rotation.set(0, Math.PI / 2, 0);
    plane6.position.set(35, 2, 0);
    scene.add(plane6);

    const geometry7 = new THREE.PlaneGeometry(12, 4);
    const plane7 = new THREE.Mesh(geometry7, wallMaterial);
    plane7.castShadow = true;
    plane7.receiveShadow = true;    
    plane7.rotation.set(0, Math.PI / 2, 0);
    plane7.position.set(35, 2, 22);
    scene.add(plane7);

    const geometry8 = new THREE.PlaneGeometry(12, 4);
    const plane8 = new THREE.Mesh(geometry8, wallMaterial);
    plane8.castShadow = true;
    plane8.receiveShadow = true;    
    plane8.rotation.set(0, Math.PI / 2, 0);
    plane8.position.set(35, 2, 44);
    scene.add(plane8);

    const geometry9 = new THREE.PlaneGeometry(100, 15);
    const plane9 = new THREE.Mesh(geometry9, floorMaterial);
    plane9.castShadow = true;
    plane9.receiveShadow = true;    
    plane9.rotation.set(Math.PI / 2, 0, Math.PI / 2);
    plane9.position.set(42.5, 0.05, 0);
    scene.add(plane9);

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push({
            id : i,
            x: getPostition(27, 48),
            y: 2,
            z: getPostition(-39, 39),
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
            maxSpeed: maxSpeed,
            maxForce: maxForce,
            horizon: HORIZON,
            k: k,
            fx: 0,
            fz: 0,
            yield: false,
            group: 1,
        });
    }

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push({
            id : i + COUNT,
            x: getPostition(0, 23),
            y: 2,
            z: getPostition(-39, 39),
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
            maxSpeed: maxSpeed,
            maxForce: maxForce,
            horizon: HORIZON,
            k: k,
            fx: 0,
            fz: 0,
            yield: true,
            group: 2,
        });
    }

    let agentGeometry, agent;

    for (let i = 0; i < COUNT; i++) {
        agentGeometry = new THREE.CylinderGeometry(agents[i].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group1Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        agents[i].agent = agent;

        agentGeometry = new THREE.CylinderGeometry(agents[i + COUNT].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group2Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        agents[i + COUNT].agent = agent;
    }

    console.log(agents.length);
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
        member.agent.material = member.group == 1 ? group1Mat : group2Mat;

        if (member.z > -50 && member.z < - 17) {
            member.tz = -28;
        } else if (member.z > -17 && member.z < 16) {
            member.tz = 0;
        } else if (member.z > 16 && member.z < 49) {
            member.tz = 28;
        }

        if (member.x > 25) {
            if ((member.z > -50 && member.z < -34) || (member.z > -22 && member.z < -6) || (member.z > 6 && member.z < 22) || (member.z > 34 && member.z < 50)) {
                if (member.x < 25 + RADIUS + 0.1) {
                    member.x = 25 + RADIUS + 0.1;
                }
            }
        } else if (member.x < 25) {
            if ((member.z > -50 && member.z < -34) || (member.z > -22 && member.z < -6) || (member.z > 6 && member.z < 22) || (member.z > 34 && member.z < 50)) {
                if (member.x > 25 - RADIUS - 0.1) {
                    member.x = 25 - RADIUS - 0.1;
                }
            }
        }

        if (member.z > 50 - RADIUS) {
            member.z = 50 - RADIUS;
        }

        if (member.group == 1) {
            if (member.x > 25) {
                member.tx = 23;
                member.fx = getPostition(-45, 0);
                member.fz = getPostition(-39, 39);
            } else {
                member.tx = member.fx;
                member.tz = member.fz;
            }
        } else {
            if (member.x < 25) {
                member.tx = 27;
                member.fx = getPostition(27, 48);
                member.fz = getPostition(-39, 39);
            } else {
                member.tx = member.fx;
                member.tz = member.fz;
            }
        }


        PHYSICS.update(member, agents);
    });

    renderer.render(scene, camera);
};

animate();