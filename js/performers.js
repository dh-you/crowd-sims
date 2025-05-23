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

let pickableObjects = [];
let selected = null;
let mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

const COUNT = 100;
const RADIUS = 1;
const MAXSPEED = 7.5;
const COMFORT = 15;

let performer = {
    id : -1,
    x: 0,
    y: 2,
    z: 45,
    vx: 0,
    vy: 0,
    vz: 0,
    gx: 0,
    gy: 0,
    gz: 0,
    tx: 0,
    ty: 2,
    tz: 45,
    radius: RADIUS,
    maxSpeed: MAXSPEED,
    isWatching: false,
};
let agents = [performer];

const pedestrianMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const onlookerMat = new THREE.MeshLambertMaterial({
    color: 0x0000ff
});
const performerMat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

init();
render();

function getPostition() {
    return [Math.random() * 90 - 45, Math.random() * 30 - 15];
}

function getVelocity() {
    return Math.random() - 0.5;
}

function generateViewingPosition() {
    let theta = Math.random() * Math.PI + Math.PI;
    return [performer.x + COMFORT * Math.cos(theta), performer.z + COMFORT * Math.sin(theta)];
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
    const texture = loader.load('resources/OIP.jpg');
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

    // street 
    const streetMaterial = new THREE.MeshBasicMaterial({color: 0x222222, side: THREE.DoubleSide});
    const streetGeometry = new THREE.PlaneGeometry(100, 30);
    const streetPlane = new THREE.Mesh(streetGeometry, streetMaterial);
    streetPlane.castShadow = true;
    streetPlane.receiveShadow = true;
    streetPlane.rotation.set(Math.PI / 2, 0, 0);
    streetPlane.position.set(0, 0.05, 0);
    scene.add(streetPlane);

    // wall
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x231709, side: THREE.DoubleSide});
    const wallGeometry = new THREE.PlaneGeometry(100, 10);
    const wallPlane = new THREE.Mesh(wallGeometry, wallMaterial);
    wallPlane.receiveShadow = true;
    wallPlane.rotation.set(Math.PI, 0, 0);
    wallPlane.position.set(0, 5, 50);
    scene.add(wallPlane);

    let agentGeometry, agent;
    for (let i = 1; i < COUNT; i++) {
        let vx = getVelocity();
        let vy = getVelocity();
        let vz = getVelocity();

        let pos = getPostition();

        agents.push({
            id : i,
            x: pos[0],
            y: 2,
            z: pos[1],
            vx: vx,
            vy: vy,
            vz: vz,
            gx: vx,
            gy: vy,
            gz: vz,
            tx: 50 * (Math.random() < 0.5 ? -1 : 1),
            ty: 2,
            tz: pos[1],
            radius: RADIUS,
            maxSpeed: MAXSPEED,
            isWatching: false,
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
    agent = new THREE.Mesh(agentGeometry, pedestrianMat);
    agent.castShadow = true;
    agent.receiveShadow = true;
    agent.userData = {
        "id": agents[0].id
    };
    scene.add(agent);
    agents[0].agent = agent;

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
        if (selected != null && member.id == selected) {
            member.isWatching = true;
            [member.tx, member.tz] = generateViewingPosition();
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