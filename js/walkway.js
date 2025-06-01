import { createScene } from './environment.js';
import { Agent } from './agent.js';
import * as THREE from 'three';
import * as PHYSICS from 'physics';

const LENGTH = 100;

let agents = [];
const COUNT = 125;
const RADIUS = 1;
const MAXSPEED = 7.5;
const HORIZON = 50;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

const { renderer, scene, camera } = createScene();
init();
render();

function getPostition() {
    return [Math.random() * 20 - 50, Math.random() * 100 - 50];
}

function getVelocity() {
    let theta = Math.random() * Math.PI * 2;
    let speed = Math.random() * MAXSPEED;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

function init() {
    // walls 
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x222222, side: THREE.DoubleSide});
    const streetMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, side: THREE.DoubleSide });

    const geometry10 = new THREE.PlaneGeometry(60, 2);
    const plane10 = new THREE.Mesh(geometry10, wallMaterial);
    plane10.rotation.set(Math.PI, 0, 0);
    plane10.position.set(0, 1, 15);
    scene.add(plane10);

    const geometry11 = new THREE.PlaneGeometry(60, 2);
    const plane11 = new THREE.Mesh(geometry11, wallMaterial);
    plane11.rotation.set(Math.PI, 0, 0);
    plane11.position.set(0, 1, 10);
    scene.add(plane11);

    const geometry12 = new THREE.PlaneGeometry(60, 5);
    const plane12 = new THREE.Mesh(geometry12, streetMaterial);
    plane12.castShadow = true;
    plane12.receiveShadow = true;
    plane12.rotation.set(Math.PI / 2, 0, 0);
    plane12.position.set(0, 0.05, 12.5);
    scene.add(plane12);

    const geometry13 = new THREE.PlaneGeometry(60, 2);
    const plane13 = new THREE.Mesh(geometry13, wallMaterial);
    plane13.rotation.set(Math.PI, 0, 0);
    plane13.position.set(0, 1, -10);
    scene.add(plane13);

    const geometry14 = new THREE.PlaneGeometry(60, 2);
    const plane14 = new THREE.Mesh(geometry14, wallMaterial);
    plane14.rotation.set(Math.PI, 0, 0);
    plane14.position.set(0, 1, -15);
    scene.add(plane14);

    const geometry15 = new THREE.PlaneGeometry(60, 5);
    const plane15 = new THREE.Mesh(geometry15, streetMaterial);
    plane15.castShadow = true;
    plane15.receiveShadow = true;
    plane15.rotation.set(Math.PI / 2, 0, 0);
    plane15.position.set(0, 0.05, -12.5);
    scene.add(plane15);

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();
        let pos = getPostition();

        let k = 1.5 + Math.random() * 1.5;
        let maxSpeed = Math.random() * (MAXSPEED - 5) + 5;
        let maxForce = 30 + Math.random() * 40;  

        agents.push(new Agent(
            pos[0], 2, pos[1],            
            v[0], 0, v[1],
            0, 0, 0,
            0, 2, pos[1],            
            RADIUS, maxSpeed, maxForce, HORIZON, false, 3.0, k
        ));

        agents[i].setData("id", i);
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
        member.setData("agent", agent);
    });
}

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);

    agents.forEach(function(member) {
        member.getData("agent").position.copy(member.position);
        member.getData("agent").material = agentMat;

        if (member.position.z > LENGTH / 2 - RADIUS) {
            member.position.z = LENGTH / 2 - RADIUS;
        } else if (member.position.z < -LENGTH / 2 + RADIUS) {
            member.position.z = -LENGTH / 2 + RADIUS;
        }

        if (member.position.z < 0) {
            member.target.z = -12.5;
        } else {
            member.target.z = 12.5;
        }

        member.target.x = -32.5;
        member.maxSpeed = MAXSPEED;

        if (member.position.x >= -30 && member.position.x <= 30) {
            if (member.position.z < 0) {
                if (member.position.z > -15 && member.position.z < -10) {
                    member.maxSpeed = MAXSPEED * 1.5;

                    if (member.position.z < -15 + RADIUS) {
                        member.position.z = -15 + RADIUS;
                    } else if (member.position.z > -10 - RADIUS) {
                        member.position.z = -10 - RADIUS;
                    }

                } else {
                    if (member.position.z < -12.5 && member.position.z > -15 - RADIUS) {
                        member.position.z = -15 - RADIUS;
                    } else if (member.position.z > -12.5 && member.position.z < -10 + RADIUS) {
                        member.position.z = -10 + RADIUS;
                    }
                }
            } else {
                if (member.position.z > 10 && member.position.z < 15) {
                    member.maxSpeed = MAXSPEED * 1.5;

                    if (member.position.z < 10 + RADIUS) {
                        member.position.z = 10 + RADIUS;
                    } else if (member.position.z > 15 - RADIUS) {
                        member.position.z = 15 - RADIUS;
                    }

                } else {
                    if (member.position.z > 12.5 && member.position.z < 15 + RADIUS) {
                        member.position.z = 15 + RADIUS;
                    } else if (member.position.z < 12.5 && member.position.z > 10 - RADIUS) {
                        member.position.z = 10 - RADIUS;
                    }
                }
            }
        }

        if (Math.abs(member.position.x + 32.5) <= 7.5 && member.position.z > -15 && member.position.z < -10) {
            member.target.x = 200;
        } else if (Math.abs(member.position.x + 32.5) <= 7.5 && member.position.z > 10 && member.position.z < 15) {
            member.target.x = 200;
        }

        if (member.position.x >= -30 && member.position.z > -15 && member.position.z < -10) {
            member.target.x = 200;
        } else if (member.position.x >= -30 && member.position.z > 10 && member.position.z < 15) {
            member.target.x = 200;
        }

        if (member.position.x >= 25) {
            member.target.x = 200;
        }

        PHYSICS.update(member, agents);
    });

    renderer.render(scene, camera);
};

animate();