import * as THREE from 'three';

const thickness = 0.5;
const wallMaterial = new THREE.MeshStandardMaterial({
    color: 0xe3e3e3,
    metalness: 0.4,
    roughness: 0.3
});

export class Wall {
    constructor(width, height, vertical, position) {
        const geometry = vertical ? new THREE.BoxGeometry(width, height, thickness) : new THREE.BoxGeometry(thickness, height, width);
        this.mesh = new THREE.Mesh(geometry, wallMaterial);
        this.mesh.position.copy(position);
        this.width = width;
        this.height = height;
        this.thickness = thickness;
        this.vertical = vertical;
        this.box = new THREE.Box3().setFromObject(this.mesh);
    }

    collisionResolve(agent) {
        agent.sphere.set(agent.position.clone(), agent.radius);

        if (this.box.intersectsSphere(agent.sphere)) {
            // get collision direction
            const closest = this.box.clampPoint(agent.position, new THREE.Vector3());
            const push = agent.position.clone().sub(closest);
            const dist = push.length();
            push.normalize();

            // push agent outwards
            const penetration = agent.radius - dist;
            agent.position.add(push.multiplyScalar(penetration));
        }
    }
}