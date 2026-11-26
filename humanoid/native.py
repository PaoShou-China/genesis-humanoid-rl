"""Version-pinned Genesis accessors kept out of task/control logic."""


def effective_link_masses(entity):
    """Include runtime mass shifts, which get_links_inertial_mass() omits."""
    nominal = entity.get_links_inertial_mass()
    indices = [link.idx for link in entity.links]
    shift = entity.solver.get_links_mass_shift(indices)
    return nominal + shift
