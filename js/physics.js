import * as UTILS from './utils.js'
import * as THREE from 'three';

export function timeToCollision(agent, neighbor) {
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

export function applyForce(agent, f, timestep) {
    const force = f.length();
    if (force > agent.maxForce) {
        f.x = agent.maxForce * f.x / force;
        f.z = agent.maxForce * f.z / force;
    }

    agent.velocity.x += f.x * timestep;
    agent.velocity.z += f.z * timestep;

    const speed = agent.velocity.length();
    if (speed > agent.maxSpeed) {
        agent.velocity.x = agent.maxSpeed * agent.velocity.x / speed;
        agent.velocity.z = agent.maxSpeed * agent.velocity.z / speed;
    }

    agent.position.x += agent.velocity.x * timestep;
    agent.position.z += agent.velocity.z * timestep;
}

export function updateAgents(agent, agents, timestep) {
    agent.goal.x = agent.target.x - agent.position.x;
    agent.goal.z = agent.target.z - agent.position.z;

    const fxGoal = agent.k * (agent.goal.x - agent.velocity.x);
    const fzGoal = agent.k * (agent.goal.z - agent.velocity.z);

    let fxAvoid = 0;
    let fzAvoid = 0;

    let fxSidestep = 0;
    let fzSidestep = 0;

    agents.forEach(function(neighbor) {
        if (neighbor.id != agent.id) {
            const t = timeToCollision(agent, neighbor);

            // predicted direction vector in future
            const dir = new THREE.Vector3(
                agent.position.x - neighbor.position.x,
                0,
                agent.position.z - neighbor.position.z
            ).normalize();

            // global left and right directions, 90 deg rotation from up vector
            const left = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), Math.PI / 2);
            const right = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), -Math.PI / 2);
            
            // is neighbor going left or right
            const leftDot = left.dot(neighbor.velocity);
            const rightDot = right.dot(neighbor.velocity);

            // side step in opposite direction of other neighbor
            let sidestep;
            if (leftDot < 0 && rightDot < 0) { // neighbor is neither going left or right
                sidestep = leftDot < rightDot ? left : right;
            } else { // neighbor is aligned with either left or right
                sidestep = leftDot < 0 ? left : right;
            }

            if (t >= 0 && t <= agent.horizon) {
                let avoidanceWeight = (agent.horizon - t) / (t + UTILS.EPSILON);

                fxAvoid += agent.avoid * dir.x * avoidanceWeight;
                fzAvoid += agent.avoid * dir.z * avoidanceWeight;

                fxSidestep += agent.sidestep * sidestep.x * avoidanceWeight;
                fzSidestep += agent.sidestep * sidestep.z * avoidanceWeight;
            }
        }
    });

    const fx = fxGoal + fxAvoid + fxSidestep;
    const fz = fzGoal + fzAvoid + fzSidestep;

    applyForce(agent, new THREE.Vector3(fx, 0, fz), timestep);
}