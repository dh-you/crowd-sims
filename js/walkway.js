import * as THREE from 'three';
import * as PHYSICS from 'physics';
import {
    OrbitControls
} from "three/addons/controls/OrbitControls.js";

const LENGTH = 100;

let renderer, scene, camera;
let world = {
    x: LENGTH,
    z: LENGTH
};

let agents = [];
const COUNT = 125;
const RADIUS = 1;
const MAXSPEED = 7.5;
const MAXFORCE = 75;
const HORIZON = 50;
const K = 2;

const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

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
    // renderer
    renderer = new THREE.WebGLRenderer();
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; //
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    // scene
    scene = new THREE.Scene();
    // camera
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);

    camera.position.set(-67.26, 54.07, -3.77);
    camera.rotation.order = 'YXZ';
    camera.rotation.y = -1.6267;
    camera.rotation.x = -0.46;

    // controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener('change', render);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.maxPolarAngle = Math.PI / 2;

    // light
    const light = new THREE.PointLight(0xffffff, 0.9, 0, 100000);
    light.position.set(0, 50, 120);
    light.castShadow = true;
    light.shadow.mapSize.width = 512; // default
    light.shadow.mapSize.height = 512; // default
    light.shadow.camera.near = 0.5; // default
    light.shadow.camera.far = 5000; // default

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.castShadow = true;
    directionalLight.position.set(-5, 20, 4);
    directionalLight.target.position.set(9, 0, -9);
    directionalLight.shadow.camera.left *= 9;
    directionalLight.shadow.camera.right *= 9;
    directionalLight.shadow.camera.top *= 9;
    directionalLight.shadow.camera.bottom *= 9;

    scene.add(directionalLight);
    const cameraHelper = new THREE.CameraHelper(directionalLight.shadow.camera);

    // axes
    // scene.add(new THREE.AxesHelper(40));
    const loader = new THREE.TextureLoader();
    const texture = loader.load('../resources/OIP.jpg');
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.NearestFilter;
    const repeats = 40 / 32;
    texture.repeat.set(repeats, repeats);

    // grid
    const geometry = new THREE.PlaneGeometry(world.x, world.z, 10, 10);
    //const material = new THREE.MeshBasicMaterial( {  opacity: 0.5, transparent: true } );
    const material = new THREE.MeshPhongMaterial({
        map: texture,
        //side: THREE.DoubleSide,
    });
    const grid = new THREE.Mesh(geometry, material);
    grid.castShadow = true; //default is false
    grid.receiveShadow = true; //default  
    grid.rotation.order = 'YXZ';
    grid.rotation.y = -Math.PI / 2;
    grid.rotation.x = -Math.PI / 2;
    scene.add(grid);

    // walls 
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x222222, side: THREE.DoubleSide});
    const walkwayMaterial = new THREE.MeshBasicMaterial({color: 0x000000, side: THREE.DoubleSide});

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
    const plane12 = new THREE.Mesh(geometry12, walkwayMaterial);
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
    const plane15 = new THREE.Mesh(geometry15, walkwayMaterial);
    plane15.castShadow = true;
    plane15.receiveShadow = true;
    plane15.rotation.set(Math.PI / 2, 0, 0);
    plane15.position.set(0, 0.05, -12.5);
    scene.add(plane15);

    for (let i = 0; i < COUNT; i++) {
        let v = getVelocity();
        let pos = getPostition();

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
            tx: 0,
            ty: 2,
            tz: pos[1],
            radius: RADIUS,
            maxSpeed: MAXSPEED,
            horizon: HORIZON,
            k: K,
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

        if (member.z > LENGTH / 2 - RADIUS) {
            member.z = LENGTH / 2 - RADIUS;
        } else if (member.z < -LENGTH / 2 + RADIUS) {
            member.z = -LENGTH / 2 + RADIUS;
        }

        if (member.z < 0) {
            member.tz = -12.5;
        } else {
            member.tz = 12.5;
        }

        member.tx = -32.5;
        member.maxSpeed = MAXSPEED;

        if (member.x >= -30 && member.x <= 30) {
            if (member.z < 0) {
                if (member.z > -15 && member.z < -10) {
                    member.maxSpeed = MAXSPEED * 1.5;

                    if (member.z < -15 + RADIUS) {
                        member.z = -15 + RADIUS;
                    } else if (member.z > -10 - RADIUS) {
                        member.z = -10 - RADIUS;
                    }
                    
                } else {
                    if (member.z < -12.5 && member.z > -15 - RADIUS) {
                        member.z = -15 - RADIUS;
                    } else if (member.z > -12.5 && member.z < -10 + RADIUS) {
                        member.z = -10 + RADIUS;
                    }
                } 
            } else {
                if (member.z > 10 && member.z < 15) {
                    member.maxSpeed = MAXSPEED * 1.5;

                    if (member.z < 10 + RADIUS) {
                        member.z = 10 + RADIUS;
                    } else if (member.z > 15 - RADIUS) {
                        member.z = 15 - RADIUS;
                    }

                } else {
                    if (member.z > 12.5 && member.z < 15 + RADIUS) {
                        member.z = 15 + RADIUS;
                    } else if (member.z < 12.5 && member.z > 10 - RADIUS) {
                        member.z = 10 - RADIUS;
                    }
                }
            }
        }

        if (Math.abs(member.x + 32.5) <= 7.5 && member.z > -15 && member.z < -10) {
            member.tx = 200;
        } else if (Math.abs(member.x + 32.5) <= 7.5 && member.z > 10 && member.z < 15) {
            member.tx = 200;
        }

        if (member.x >= -30 && member.z > -15 && member.z < -10) {
            member.tx = 200;
        } else if (member.x >= -30 && member.z > 10 && member.z < 15) {
            member.tx = 200;
        }

        if (member.x >= 25) {
            member.tx = 200;
        }

        PHYSICS.update(member, agents);
    });

    renderer.render(scene, camera);
};

animate();