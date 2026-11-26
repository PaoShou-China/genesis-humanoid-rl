"""Stable task IDs; task definitions are independent of controller choice."""

TASK_DEFINITIONS = {
    "H1": ("h1_stiff_pd", "Standing balance", None),
    "H2": ("h2_velocity", "Velocity tracking", None),
    "H3": ("h3_stiff_pd", "Push recovery", None),
    "H4": ("h4_terrain", "Mixed rough terrain", None),
    "H5": ("h5_motion", "Reference joint-motion tracking", "motion"),
    "H6": ("h6_heading", "Heading control", "heading"),
    "H7": ("h7_waypoint", "Waypoint reaching", "waypoint"),
    "H8": ("h8_height", "Crouch / base-height tracking", "height"),
    "H9": ("h9_reach", "Bimanual reaching", "reach"),
    "H10": ("h10_getup", "Prone / supine recovery", "getup"),
    "H11": ("h11_jump", "Jump and landing", "jump"),
    "H12": ("h12_stairs", "Stair locomotion", None),
    "H13": ("h13_slope", "Slope locomotion", None),
    "H14": ("h14_push_object", "Contact-based object pushing", "object_push"),
}
TASKS = {key: value[0] for key, value in TASK_DEFINITIONS.items()}
CUSTOM_TASKS = {key: value[2] for key, value in TASK_DEFINITIONS.items() if value[2]}
TERRAIN_TASKS = {"H4", "H12", "H13"}
