"""World-frame force pulses on the torso, applied at every control step."""

import torch


def apply_push(env, spec):
    """Apply newtons at torso COM; return actual requested force per environment."""
    force = torch.zeros(env.num_envs, 3, device=env.device)
    if spec:
        start, duration = (round(spec[k] / env.step_dt) for k in ("start_s", "duration_s"))
        if duration < 1 or any(
            abs(spec[k] / env.step_dt - round(spec[k] / env.step_dt)) > 1e-5
            for k in ("start_s", "duration_s")
        ):
            raise ValueError("Push timing must be an integer multiple of the control period")
        active = (env.episode_length_buf >= start) & (env.episode_length_buf < start + duration)
        force[active] = torch.tensor(spec["force_n"], device=env.device)
        entity = env.scene["robot"].gs_entity
        link = entity.get_link("torso_link")
        # Genesis exposes link forces on its native rigid solver, not the entity wrapper.
        entity.solver.apply_links_external_force(force[:, None, :], [link.idx], ref="link_com")
    return force
