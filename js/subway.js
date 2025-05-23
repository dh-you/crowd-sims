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
let group1 = [];
let group2 = [];
const COUNT = 20;
const RADIUS = 1;
const MAXSPEED = 5;
const group1Mat = new THREE.MeshLambertMaterial({
    color: 0x00ff00
});
const group2Mat = new THREE.MeshLambertMaterial({
    color: 0xff0000
});

init();
render();

function getPostition(min, max) {
    return Math.random() * (max - min) + min;
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

    // walls 
    const wallMaterial = new THREE.MeshBasicMaterial({color: 0x222222, side: THREE.DoubleSide});

    const geometry1 = new THREE.PlaneGeometry(100, 4);
    const plane1 = new THREE.Mesh(geometry1, wallMaterial);
    plane1.rotation.set(0, Math.PI / 2, 0);
    plane1.position.set(50, 2, 0);
    scene.add(plane1);

    const geometry2 = new THREE.PlaneGeometry(25, 4);
    const plane2 = new THREE.Mesh(geometry2, wallMaterial);
    plane2.rotation.set(Math.PI, 0, 0);
    plane2.position.set(37.5, 2, -50);
    scene.add(plane2);

    const geometry3 = new THREE.PlaneGeometry(25, 4);
    const plane3 = new THREE.Mesh(geometry3, wallMaterial);
    plane3.rotation.set(Math.PI, 0, 0);
    plane3.position.set(37.5, 2, 50);
    scene.add(plane3);

    const geometry4 = new THREE.PlaneGeometry(16, 4);
    const plane4 = new THREE.Mesh(geometry4, wallMaterial);
    plane4.rotation.set(0, Math.PI / 2, 0);
    plane4.position.set(25, 2, -42);
    scene.add(plane4);

    const geometry5 = new THREE.PlaneGeometry(16, 4);
    const plane5 = new THREE.Mesh(geometry5, wallMaterial);
    plane5.rotation.set(0, Math.PI / 2, 0);
    plane5.position.set(25, 2, -14);
    scene.add(plane5);

    const geometry6 = new THREE.PlaneGeometry(16, 4);
    const plane6 = new THREE.Mesh(geometry6, wallMaterial);
    plane6.rotation.set(0, Math.PI / 2, 0);
    plane6.position.set(25, 2, 14);
    scene.add(plane6);

    const geometry7 = new THREE.PlaneGeometry(16, 4);
    const plane7 = new THREE.Mesh(geometry7, wallMaterial);
    plane7.rotation.set(0, Math.PI / 2, 0);
    plane7.position.set(25, 2, 42);
    scene.add(plane7);

    for (let i = 0; i < COUNT; i++) {
        let vx = getVelocity();
        let vy = getVelocity();
        let vz = getVelocity();

        group1.push({
            id : i,
            x: getPostition(27, 48),
            y: 2,
            z: getPostition(-39, 39),
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
            fx: 0,
            fz: 0,
        })

        group2.push({
            id : i,
            x: getPostition(0, 23),
            y: 2,
            z: getPostition(-39, 39),
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
            fx: 0,
            fz: 0,
        })

        agents.push(group1[i], group2[i]);
    }

    let agentGeometry, agent;

    for (let i = 0; i < COUNT; i++) {
        agentGeometry = new THREE.CylinderGeometry(group1[i].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group1Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        group1[i].agent = agent;

        agentGeometry = new THREE.CylinderGeometry(group2[i].radius, 1, 4, 16);
        agent = new THREE.Mesh(agentGeometry, group2Mat);
        agent.castShadow = true;
        agent.receiveShadow = true;
        scene.add(agent);
        group2[i].agent = agent;
    }
}

function render() {
    renderer.render(scene, camera);
}

function animate() {
    requestAnimationFrame(animate);

    group1.forEach(function(member) {
        member.agent.position.x = member.x;
        member.agent.position.y = member.y;
        member.agent.position.z = member.z;
        member.agent.material = group1Mat;

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

        if (member.x > 25) {
            member.tx = 23;
            member.fx = getPostition(-45, 0);
            member.fz = getPostition(-39, 39);
        } else {
            member.tx = member.fx;
            member.tz = member.fz;
        }

        PHYSICS.update(member, agents);
    });

    group2.forEach(function(member) {
        member.agent.position.x = member.x;
        member.agent.position.y = member.y;
        member.agent.position.z = member.z;
        member.agent.material = group2Mat;

        if (member.z > -50 && member.z < - 17) {
            member.tz = -28;
        } else if (member.z > -17 && member.z < 16) {
            member.tz = 0;
        } else if (member.z > 16 && member.z < 49) {
            member.tz = 28;
        }

        if (member.x < 25) {
            if ((member.z > -50 && member.z < -34) || (member.z > -22 && member.z < -6) || (member.z > 6 && member.z < 22) || (member.z > 34 && member.z < 50)) {
                if (member.x > 25 - RADIUS - 0.1) {
                    member.x = 25 - RADIUS - 0.1;
                }
            }
        } else if (member.x > 25) {
            if ((member.z > -50 && member.z < -34) || (member.z > -22 && member.z < -6) || (member.z > 6 && member.z < 22) || (member.z > 34 && member.z < 50)) {
                if (member.x < 25 + RADIUS + 0.1) {
                    member.x = 25 + RADIUS + 0.1;
                }
            }
        }

        if (member.x < 25) {
            member.tx = 27;
            member.fx = getPostition(27, 48);
            member.fz = getPostition(-39, 39);
        } else {
            member.tx = member.fx;
            member.tz = member.fz;
        }

        PHYSICS.update(member, agents);
    });

    renderer.render(scene, camera);
};

animate();