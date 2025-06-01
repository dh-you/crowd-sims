import * as THREE from 'three';

const TIMESTEP = 0.005;
const EPSILON = 0.001;

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
        if (neighbor.getData("id") != agent.getData("id")) {
            let t = timeToCollision(agent, neighbor);

            let dir = new THREE.Vector3(
                (agent.position.x + agent.velocity.x * t) - (neighbor.position.x + neighbor.velocity.x * t),
                0,
                (agent.position.z + agent.velocity.z * t) - (neighbor.position.z + neighbor.velocity.z * t)
            );

            dir.normalize();
        
            if (t >= 0 && t <= agent.horizon) {
                fxAvoid += (agent.yieldStatus ? agent.yieldFactor : 1) * dir.x * (agent.horizon - t) / (t + EPSILON);
                fzAvoid += (agent.yieldStatus ? agent.yieldFactor : 1) * dir.z * (agent.horizon - t) / (t + EPSILON);
            }
        }
    });

    let fx = fxGoal + fxAvoid;
    let fz = fzGoal + fzAvoid;

    applyForce(agent, new THREE.Vector3(fx, 0, fz));
}