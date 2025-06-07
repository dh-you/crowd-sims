import * as THREE from 'three';

const TIMESTEP = 0.005;
const EPSILON = 0.001;
const sideStepStrength = 0.5;

function timeToCollision(agent, neighbor) {
    let r = agent.radius + neighbor.radius;
    let w = new THREE.Vector3(neighbor.position.x - agent.position.x, 0, neighbor.position.z - agent.position.z);
    
    let c =  w.dot(w) - r * r;
    if (c < 0) { return 0; }

    let v = new THREE.Vector3(agent.velocity.x - neighbor.velocity.x, 0, agent.velocity.z - neighbor.velocity.z);

    let a = v.dot(v);
    let b = w.dot(v);

    let discriminant = b * b - a * c;
    if (discriminant <= 0) { return Infinity; }

    let tau = (b - Math.sqrt(discriminant)) / a;
    if (tau < 0) { return Infinity; }

    return tau;
}

export function applyForce(agent, f) {
    let force = f.length();
    if (force > agent.maxForce) {
        f.x = agent.maxForce * f.x / force;
        f.z = agent.maxForce * f.z / force;
    }

    agent.velocity.x += f.x * TIMESTEP;
    agent.velocity.z += f.z * TIMESTEP;

    let speed = agent.velocity.length();
    if (speed > agent.maxSpeed) {
        agent.velocity.x = agent.maxSpeed * agent.velocity.x / speed;
        agent.velocity.z = agent.maxSpeed * agent.velocity.z / speed;
    }

    agent.position.x += agent.velocity.x * TIMESTEP;
    agent.position.z += agent.velocity.z * TIMESTEP;
}

export function update(agent, agents) {
    agent.goal.x = agent.target.x - agent.position.x;
    agent.goal.z = agent.target.z - agent.position.z;

    let fxGoal = agent.k * (agent.goal.x - agent.velocity.x);
    let fzGoal = agent.k * (agent.goal.z - agent.velocity.z);

    let fxAvoid = 0;
    let fzAvoid = 0;

    agents.forEach(function(neighbor) {
        if (neighbor.id != agent.id) {
            let t = timeToCollision(agent, neighbor);

            let dir = new THREE.Vector3(
                (agent.position.x + agent.velocity.x * t) - (neighbor.position.x + neighbor.velocity.x * t),
                0,
                (agent.position.z + agent.velocity.z * t) - (neighbor.position.z + neighbor.velocity.z * t)
            ).normalize();

            let left = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), Math.PI / 2);
            let right = dir.clone().applyAxisAngle(new THREE.Vector3(0,1,0), -Math.PI / 2);
            
            let leftDot = left.dot(neighbor.velocity);
            let rightDot = right.dot(neighbor.velocity);

            let sidestep;
            if (leftDot < 0 && rightDot < 0) {
                sidestep = leftDot < rightDot ? left : right;
            } else if (leftDot > 0 && rightDot > 0) {
                sidestep = leftDot < rightDot ? right : left;
            } else {
                sidestep = leftDot < 0 ? left : right;
            }

            if (t >= 0 && t <= agent.horizon) {
                fxAvoid += dir.x * (agent.horizon - t) / (t + EPSILON);
                fzAvoid += dir.z * (agent.horizon - t) / (t + EPSILON);

                fxAvoid += sideStepStrength * sidestep.x * (agent.horizon - t) / (t + EPSILON);
                fzAvoid += sideStepStrength * sidestep.z * (agent.horizon - t) / (t + EPSILON);
            }
        }
    });

    let fx = fxGoal + fxAvoid;
    let fz = fzGoal + fzAvoid;

    applyForce(agent, new THREE.Vector3(fx, 0, fz));
}