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
const COUNT = 200;
const RADIUS = 1;
const MAXSPEED = 10;
const agentMat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});

init();
render();

function getPostition() {
    return [Math.random() * 40 - 45, Math.random() * 90 - 45];
}

function getVelocity() {
    return Math.random() - 0.5;
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
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x36454F, side: THREE.DoubleSide});

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
        let vx = getVelocity();
        let vy = getVelocity();
        let vz = getVelocity();

        agents.push({
            id : i,
            x: getPostition()[1],
            y: 2,
            z: getPostition()[0],
            vx: vx,
            vy: vy,
            vz: vz,
            gx: vx,
            gy: vy,
            gz: vz,
            tx: 0,
            ty: 2,
            tz: 0,
            radius: RADIUS,
            maxSpeed: MAXSPEED,
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