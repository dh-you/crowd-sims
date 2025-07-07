import * as UTILS from './utils.js'
import * as THREE from 'three';

const TIMESTEP = 0.005;
const sideStepStrength = 0.5;

function timeToCollision(agent, neighbor) {
    const r = agent.radius + neighbor.radius;
    const w = new THREE.Vector3(neighbor.position.x - agent.position.x, 0, neighbor.position.z - agent.position.z);
    
    const c =  w.dot(w) - r * r;
    if (c < 0) { return 0; }

    const v = new THREE.Vector3(agent.velocity.x - neighbor.velocity.x, 0, agent.velocity.z - neighbor.velocity.z);

    const a = v.dot(v);
    const b = w.dot(v);

    const discriminant = b * b - a * c;
    if (discriminant <= 0) { return Infinity; }

    const tau = (b - Math.sqrt(discriminant)) / a;
    if (tau < 0) { return Infinity; }

    return tau;
}

export function applyForce(agent, f) {
    const force = f.length();
    if (force > agent.maxForce) {
        f.x = agent.maxForce * f.x / force;
        f.z = agent.maxForce * f.z / force;
    }

    agent.velocity.x += f.x * TIMESTEP;
    agent.velocity.z += f.z * TIMESTEP;

    const speed = agent.velocity.length();
    if (speed > agent.maxSpeed) {
        agent.velocity.x = agent.maxSpeed * agent.velocity.x / speed;
        agent.velocity.z = agent.maxSpeed * agent.velocity.z / speed;
    }

    agent.position.x += agent.velocity.x * TIMESTEP;
    agent.position.z += agent.velocity.z * TIMESTEP;
}

export function updateAgents(agent, agents) {
    agent.goal.x = agent.target.x - agent.position.x;
    agent.goal.z = agent.target.z - agent.position.z;

    const fxGoal = agent.k * (agent.goal.x - agent.velocity.x);
    const fzGoal = agent.k * (agent.goal.z - agent.velocity.z);

    let fxAvoid = 0;
    let fzAvoid = 0;

    agents.forEach(function(neighbor) {
        if (neighbor.id != agent.id) {
            const t = timeToCollision(agent, neighbor);

            const dir = new THREE.Vector3(
                (agent.position.x + agent.velocity.x * t) - (neighbor.position.x + neighbor.velocity.x * t),
                0,
                (agent.position.z + agent.velocity.z * t) - (neighbor.position.z + neighbor.velocity.z * t)
            ).normalize();

            const left = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), Math.PI / 2);
            const right = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), -Math.PI / 2);
            
            const leftDot = left.dot(neighbor.velocity);
            const rightDot = right.dot(neighbor.velocity);

            // side step in opposite direction of other neighbor
            let sidestep;
            if (leftDot < 0 && rightDot < 0) {
                sidestep = leftDot < rightDot ? left : right;
            } else if (leftDot > 0 && rightDot > 0) {
                sidestep = leftDot < rightDot ? right : left;
            } else {
                sidestep = leftDot < 0 ? left : right;
            }

            if (t >= 0 && t <= agent.horizon) {
                fxAvoid += dir.x * (agent.horizon - t) / (t + UTILS.EPSILON);
                fzAvoid += dir.z * (agent.horizon - t) / (t + UTILS.EPSILON);

                fxAvoid += sideStepStrength * sidestep.x * (agent.horizon - t) / (t + UTILS.EPSILON);
                fzAvoid += sideStepStrength * sidestep.z * (agent.horizon - t) / (t + UTILS.EPSILON);
            }
        }
    });

    const fx = fxGoal + fxAvoid;
    const fz = fzGoal + fzAvoid;

    applyForce(agent, new THREE.Vector3(fx, 0, fz));
}