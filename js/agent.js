import * as THREE from 'three';

export class Agent {
    constructor(id, x, y, z, vx, vy, vz, gx, gy, gz, tx, ty, tz, radius, maxSpeed, maxForce, horizon, k) {
        this.id = id;
        this.position = new THREE.Vector3(x, y, z);
        this.velocity = new THREE.Vector3(vx, vy, vz);
        this.goal = new THREE.Vector3(gx, gy, gz);
        this.target = new THREE.Vector3(tx, ty, tz);
        this.radius = radius;
        this.maxSpeed = maxSpeed;
        this.maxForce = maxForce;
        this.horizon = horizon;
        this.k = k;
        this.userData = {};
        this.sphere = new THREE.Sphere(new THREE.Vector3(), radius);
    }

    setData(label, data) {
        this.userData[label] = data;
    }

    getData(label) {
        return this.userData[label]
    }
}