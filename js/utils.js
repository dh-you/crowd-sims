export const EPSILON = 0.005;

export function getVelocity(maxSpeed) {
    const theta = Math.random() * Math.PI * 2;
    const speed = Math.random() * maxSpeed;

    return [speed * Math.cos(theta), speed * Math.sin(theta)];
}

export function getPosition(minX, maxX, minZ, maxZ) {
    const x = Math.random() * (maxX - minX) + minX;
    const z = Math.random() * (maxZ - minZ) + minZ;

    return [x, z];
}