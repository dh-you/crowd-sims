import { LENGTH, createScene } from './environment.js';
import { Agent } from './agent.js';
import { updateAgents } from './physics.js'
import * as UTILS from './utils.js'
import * as THREE from 'three';

let agents = [];
let timestep = 0.005;

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
    TRANSITIONCAP: 2,
    wAgent: 30,
    wPerformer: 80
}

let performer = new THREE.Vector3(0, 2, 45);
let points;
let inTransition = 0;

let pickableObjects = [];

let pauseButton = document.getElementById("pause");
pauseButton.addEventListener("click", pauseButtonLogic);

let downloadButton = document.getElementById("download");
downloadButton.addEventListener("click", downloadButtonLogic);

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

const { renderer, scene, camera, world } = createScene();
init();
render();

function pauseButtonLogic() {
    console.log("pause");
    world.pause = !world.pause;
    if (!world.pause) {
        pauseButton.src = "../resources/icons/8666604_pause_icon.png";
    } else {
        pauseButton.src = "../resources/icons/8666634_play_icon.png";
    }
}

function parsePositions() {
    let result = "";
    for (let i = 1; i <= world.frame; i++) {
        let curData = world.positions[i];
        let keys = Object.keys(curData);
        keys.forEach((k, idx) => {
            let curAgent = curData[k];
            result += curAgent.x.toFixed(4) + "," + curAgent.z.toFixed(4);
            result += (idx < keys.length - 1) ? "," : "\n";
        });
    }
    return result;
}

function downloadButtonLogic() {
    console.log("download");
    if (Object.keys(world.positions).length > 0) {
        let textFile = parsePositions();
        let a = document.createElement('a');
        a.href = "data:application/octet-stream," + encodeURIComponent(textFile);
        a.download = 'trajectories.txt';
        a.click();
    }
}

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
        agents[i].setData("walking_timer", Math.random() * 10 + 5);
        console.log(agents[i].getData("walking_timer"));

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
}

function render() {
    renderer.render(scene, camera);
}

let prev = 0;
let timer = 0;

function animate(timestamp = 0) {
    requestAnimationFrame(animate);

    if (!world.pause) {
        const delta = (timestamp - prev) / 1000;
        prev = timestamp;

        timer += delta;

        agents.forEach(function (member) {
            switch (member.getData("state")) {
                case "VIEWING":
                    // subtract viewing time          
                    member.setData("viewing_timer", member.getData("viewing_timer") - (delta * timestep / 0.005));

                    if (member.getData("viewing_timer") <= 0 && inTransition < CONFIG.TRANSITIONCAP) {
                        inTransition++;
                        // move to the road
                        member.target.setComponent(2, UTILS.range(-15, 15));
                        member.setData("state", "EXITING");
                    } else if (member.getData("viewing_timer") <= 0 && inTransition >= CONFIG.TRANSITIONCAP) {
                        member.setData("viewing_timer", Math.random() * 10 + 5);
                    }
                    break;

                case "EXITING":
                    // once on the road, set target to walk to edge
                    if (member.position.distanceTo(member.target) < 5) {
                        member.horizon = CONFIG.HORIZON;
                        points.push(member.getData("viewingPosition"));
                        member.target.setComponent(0, 50 * (Math.random() < 0.5 ? -1 : 1));
                        member.setData("state", "WALKING");
                        member.setData("walking_timer", Math.random() * 10 + 5);
                        inTransition--;
                    }
                    break;

                case "WALKING":
                    // subtract walking time          
                    member.setData("walking_timer", member.getData("walking_timer") - (delta * timestep / 0.005));
                    // when walking timer expires and not too many transitioning, join the crowd
                    if (member.getData("walking_timer") <= 0 && inTransition < CONFIG.TRANSITIONCAP) {
                        inTransition++;
                        const viewingPos = generateViewingPosition(member);
                        member.setData("viewingPosition", viewingPos);
                        member.target = viewingPos.clone();
                        member.setData("state", "JOINING");
                    } else if (member.getData("walking_timer") <= 0 && inTransition >= CONFIG.TRANSITIONCAP) {
                        member.setData("walking_timer", Math.random() * 10 + 5);
                    }
                    break;

                case "JOINING":
                    member.horizon = 1;

                    // set new timer for viewer once viewing position is reached
                    if (member.position.distanceTo(member.getData("viewingPosition")) < 5) {
                        member.setData("viewing_timer", Math.random() * 10 + 5);
                        member.setData("state", "VIEWING");
                        inTransition--;
                    }
                    break;
            }
        });

        timestep = document.getElementById("timestep").value;
        document.getElementById("timestepValue").innerHTML = timestep;
        agents.forEach(function (member) {
            updateAgents(member, agents, timestep);

            // wrap around
            if (member.position.x < -LENGTH / 2 + CONFIG.RADIUS) {
                member.position.x = LENGTH / 2 - CONFIG.RADIUS;
                member.position.z *= -1;
            } else if (member.position.x > LENGTH / 2 - CONFIG.RADIUS) {
                member.position.x = -LENGTH / 2 + CONFIG.RADIUS;
                member.position.z *= -1;
            }
        });
    }
    world.frame++;
    world.positions[world.frame] = {};

    agents.forEach(function (member, index) {
        world.positions[world.frame][index] = { "x": member.position.x, "z": member.position.z, "rotation": member.getData("agent").rotation.z };

        member.getData("agent").position.copy(member.position);
    });

    renderer.render(scene, camera);
};

animate();