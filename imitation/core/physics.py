import math
from pygame.math import Vector2

EPSILON = 1e-6
OVERLAP_TOLERANCE = 0.05

def time_to_collision(agent, neighbor):
    r = agent.radius + neighbor.radius
    w = neighbor.position - agent.position
    c = w.length_squared() - r*r
    if c < 0:
        return 0
    v = agent.velocity - neighbor.velocity
    a = v.length_squared()
    b = w.dot(v)
    discriminant = b*b - a*c
    if discriminant <= 0 or a == 0:
        return math.inf
    tau = (b - math.sqrt(discriminant)) / a
    return tau if tau >= 0 else math.inf

def apply_force(agent, f, timestep):
    if f.length() > agent.max_force:
        f = f.normalize() * agent.max_force
    agent.velocity += f * timestep
    if agent.velocity.length() > agent.max_speed:
        agent.velocity = agent.velocity.normalize() * agent.max_speed
    agent.position += agent.velocity * timestep

def update_agent(agent, agents, timestep):
    agent.goal = agent.target - agent.position
    fx_goal = agent.k * (agent.goal.x - agent.velocity.x)
    fy_goal = agent.k * (agent.goal.y - agent.velocity.y)

    fx_avoid = fy_avoid = 0
    fx_sidestep = fy_sidestep = 0

    for neighbor in agents:
        if neighbor.id == agent.id:
            continue
        t = time_to_collision(agent, neighbor)
        dir_vec = agent.position - neighbor.position
        if dir_vec.length() > 0:
            dir_vec = dir_vec.normalize()
        else:
            dir_vec = Vector2(1, 0)

        left = Vector2(-dir_vec.y, dir_vec.x)
        right = Vector2(dir_vec.y, -dir_vec.x)
        sidestep = left if left.dot(neighbor.velocity) < right.dot(neighbor.velocity) else right

        if 0 <= t <= agent.horizon:
            w = (agent.horizon - t) / (t + EPSILON)
            fx_avoid += agent.avoid * dir_vec.x * w
            fy_avoid += agent.avoid * dir_vec.y * w
            fx_sidestep += agent.sidestep * sidestep.x * w
            fy_sidestep += agent.sidestep * sidestep.y * w

    fx = fx_goal + fx_avoid + fx_sidestep
    fy = fy_goal + fy_avoid + fy_sidestep
    apply_force(agent, Vector2(fx, fy), timestep)


def advance_velocity_agents(agents, actions, walls, timestep, overlap_passes=4):
    """Advance a batch of direct velocity actions without update-order bias.

    All velocities and positions are updated before collision corrections are
    considered.  Each overlap pass accumulates equal-and-opposite corrections
    from a single position snapshot, then applies them simultaneously.
    """
    if len(agents) != len(actions):
        raise ValueError("agents and actions must have the same length")

    previous_positions = [Vector2(agent.position) for agent in agents]
    for agent, action in zip(agents, actions):
        velocity = Vector2(float(action[0]), float(action[1]))
        if velocity.length() > agent.max_speed:
            velocity.scale_to_length(agent.max_speed)
        agent.velocity = velocity

    for agent in agents:
        agent.position += agent.velocity * timestep

    contact_pairs = []
    max_penetration = 0.0
    for collision_pass in range(overlap_passes):
        corrections = [Vector2(0, 0) for _ in agents]
        found_overlap = False

        for i, agent in enumerate(agents):
            for j in range(i + 1, len(agents)):
                other = agents[j]
                separation = other.position - agent.position
                distance = separation.length()
                minimum_distance = agent.radius + other.radius
                penetration = minimum_distance - distance
                if penetration <= EPSILON:
                    continue

                found_overlap = True
                if collision_pass == 0:
                    contact_pairs.append((agent.id, other.id))
                    max_penetration = max(max_penetration, penetration)

                if distance > EPSILON:
                    normal = separation / distance
                else:
                    # Head-on agents can land at exactly the same position.
                    # Their pre-integration separation supplies a geometric,
                    # order-independent normal.  If the initial state was also
                    # coincident, leave it unresolved rather than inventing an
                    # ID-based priority.
                    prior_separation = previous_positions[j] - previous_positions[i]
                    if prior_separation.length() <= EPSILON:
                        continue
                    normal = prior_separation.normalize()

                correction = normal * (penetration * 0.5)
                corrections[i] -= correction
                corrections[j] += correction

        for agent, correction in zip(agents, corrections):
            agent.position += correction

        if not found_overlap:
            break

    for agent in agents:
        for wall in walls:
            wall.collision_resolve(agent)

    # A wall projection can push an agent back into a neighbor.  Re-run the
    # same simultaneous pair correction without recording a second contact;
    # these corrections are still derived from one snapshot per pass.
    for _ in range(overlap_passes * 2):
        corrections = [Vector2(0, 0) for _ in agents]
        found_overlap = False
        for i, agent in enumerate(agents):
            for j in range(i + 1, len(agents)):
                other = agents[j]
                separation = other.position - agent.position
                distance = separation.length()
                penetration = agent.radius + other.radius - distance
                if penetration <= EPSILON:
                    continue
                found_overlap = True
                if distance > EPSILON:
                    normal = separation / distance
                else:
                    prior_separation = previous_positions[j] - previous_positions[i]
                    if prior_separation.length() <= EPSILON:
                        continue
                    normal = prior_separation.normalize()
                correction = normal * (penetration * 0.55)
                corrections[i] -= correction
                corrections[j] += correction
        for agent, correction in zip(agents, corrections):
            agent.position += correction
        if not found_overlap:
            break

    unresolved_pairs = []
    unresolved_max_penetration = 0.0
    for i, agent in enumerate(agents):
        for other in agents[i + 1:]:
            penetration = (
                agent.radius + other.radius
                - agent.position.distance_to(other.position)
            )
            if penetration > OVERLAP_TOLERANCE:
                unresolved_pairs.append((agent.id, other.id))
                unresolved_max_penetration = max(
                    unresolved_max_penetration,
                    penetration,
                )

    return {
        "contacts": len(contact_pairs),
        "contact_pairs": contact_pairs,
        "max_penetration": max_penetration,
        "unresolved_overlaps": len(unresolved_pairs),
        "unresolved_pairs": unresolved_pairs,
        "unresolved_max_penetration": unresolved_max_penetration,
    }
