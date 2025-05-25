const TIMESTEP = 0.005;
const EPSILON = 0.001;
const YIELD = 3.0;

function dot2(a, b) {
    return a[0] * b[0] + a[1] * b[1];
}

function magnitude(a) {
    return Math.sqrt(a[0]**2 + a[1]**2);
}

function timeToCollision(agent, neighbor) {
    let r = agent.radius + neighbor.radius;
    let w = [neighbor.x - agent.x, neighbor.z - agent.z];
    
    let c =  dot2(w, w) - r * r;
    if (c < 0) { return 0; }

    let v = [agent.vx - neighbor.vx, agent.vz - neighbor.vz];

    let a = dot2(v, v);
    let b = dot2(w, v);

    let discriminant = b * b - a * c;
    if (discriminant <= 0) { return Infinity; }

    let tau = (b - Math.sqrt(discriminant)) / a;
    if (tau < 0) { return Infinity; }

    return tau;
}

export function applyForce(agent, f) {
    let force = magnitude(f);
    if (force > agent.maxForce) {
        f[0] = agent.maxForce * f[0] / force;
        f[1] = agent.maxForce * f[1] / force;
    }

    agent.vx += f[0] * TIMESTEP;
    agent.vz += f[1] * TIMESTEP;

    let speed = magnitude([agent.vx, agent.vz]);
    if (speed > agent.maxSpeed) {
        agent.vx = agent.maxSpeed * agent.vx / speed;
        agent.vz = agent.maxSpeed * agent.vz / speed;
    }

    agent.x += agent.vx * TIMESTEP;
    agent.z += agent.vz * TIMESTEP;
}

export function update(agent, agents) {
    agent.gx = agent.tx - agent.x;
    agent.gz = agent.tz - agent.z;

    let fxGoal = agent.k * (agent.gx - agent.vx);
    let fzGoal = agent.k * (agent.gz - agent.vz);

    let fxAvoid = 0;
    let fzAvoid = 0;

    agents.forEach(function(neighbor) {
        if (neighbor.id != agent.id) {
            let t = timeToCollision(agent, neighbor);

            let dir = [(agent.x + agent.vx * t) - (neighbor.x + neighbor.vx * t),  (agent.z + agent.vz * t) - (neighbor.z + neighbor.vz * t)];

            if (dir[0] != 0 && dir[1] != 0) {
                dir[0] /= Math.sqrt(dot2(dir, dir));
                dir[1] /= Math.sqrt(dot2(dir, dir));
            }
        
            if (t >= 0 && t <= agent.horizon) {
                fxAvoid += (agent.yield ? YIELD : 1) * dir[0] * (agent.horizon - t) / (t + EPSILON);
                fzAvoid += (agent.yield ? YIELD : 1) * dir[1] * (agent.horizon - t) / (t + EPSILON);
            }
        }
    });

    let fx = fxGoal + fxAvoid;
    let fz = fzGoal + fzAvoid;

    applyForce(agent, [fx, fz]);
}