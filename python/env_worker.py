"""EnvWorker: one Unity player, plus the per-robot state Python keeps about it.

A thin wrapper over UnityEnvironment (step, get_steps, set_actions) and
CriticChannel (snapshots in, reset/image/pose/visual-state commands out),
carrying the dictionaries of per-(arena, robot) state -- belief clouds, hidden
states, trackers, odometers -- that persist across ticks. Deliberately has no
policy or reward logic: it is what the trainer talks to, not what it computes.
"""

try:
    from mlagents_envs.base_env import ActionTuple
    HAVE_MLAGENTS = True
except Exception:
    ActionTuple = None
    HAVE_MLAGENTS = False


class EnvWorker:
    def __init__(self, env, channel, params, behavior_name):
        self.env = env
        self.channel = channel
        self.params = params
        self.behavior_name = behavior_name

        self.z = {}
        self.image_id = {}
        self.databases = {}
        self.hidden = {}
        self.last_motor = {}
        self.last_dec_step = {}
        self.odometer = {}
        self.track_neighbor = {}
        self.belief = {}
        self.track_seed = {}
        self.pending_find_reward = {}
        self.step_count = {}
        self.traj_id = {}
        self.ep_reward = {}

    def set_num_arenas(self, n):
        self.params.set_float_parameter("num_arenas", float(n))

    def set_swarm_rng(self, seed):
        """Pin every subsequent arena respawn to an exact seed.

        Same seed, same arena -- same population count, same positions, same
        cardinal headings -- every time, including two respawns of one
        long-lived player. That exactness is the point: the test fixture reuses
        one player across a whole session, so a seed mixed with a respawn
        counter would hand two tests asking for the same seed two different
        arenas. It also means a training run must NOT leave this set, or every
        episode replays the same spawn; unity_env's own swarm_rng argument
        (KILOBOT_SWARM_RNG) is the run-level knob that varies episodes.

        None (or any negative value) means unseeded, the historical behavior.

        Takes effect on the next respawn, not immediately: SwarmManager reads it
        in SeedSwarmRng, which runs from DoReset. So set it, send_reset the
        arenas, then reset_env() to flush both in one packet. The very first
        spawn happens before any of this and can only be seeded via
        KILOBOT_SWARM_RNG.
        """
        self.params.set_float_parameter("swarm_rng", float(-1 if seed is None else seed))

    def send_poses(self, k, poses):
        self.channel.send_poses(k, poses)

    def reset_env(self):
        self.env.reset()

    def step(self):
        self.env.step()

    def get_steps(self):
        return self.env.get_steps(self.behavior_name)

    def set_actions(self, actions):
        self.env.set_actions(self.behavior_name, ActionTuple(continuous=actions))

    def snapshot(self, k):
        return self.channel.snapshot(k)

    def spawn_heading(self, k, local):
        # The communicated, immutable setup fact for whichever
        # cardinal heading this specific robot was actually given at
        # spawn, cached by actor_io.py's act() the first tick it ever
        # sees this robot (KilobotAgent.cs's own spawnHeading observation
        # field, not a live read of anything). None if that cache has no
        # entry yet for this robot -- either it hasn't appeared in a
        # decision_steps batch yet, or this is an older, not-yet-rebuilt
        # player that predates the observation carrying this column at
        # all -- callers already treat None as "fall back to the single,
        # original KNOWN_START_HEADING" when nothing is available.
        cache = getattr(self, "_spawn_heading_cache", None)
        if cache is None:
            return None
        return cache.get(k, {}).get(local)

    def send_reset(self, k, index):
        self.channel.send_reset(k, index)

    def send_image(self, k, index):
        self.channel.send_image(k, index)

    def send_robot_states(self, k, states):
        self.channel.send_robot_states(k, states)

    def pop_timing(self):
        return self.channel.pop_timing()

    def idle_other_behaviors(self):
        for name in list(self.env.behavior_specs.keys()):
            if name == self.behavior_name:
                continue
            decision_steps, _ = self.env.get_steps(name)
            spec = self.env.behavior_specs[name]
            action = spec.action_spec.empty_action(len(decision_steps))
            self.env.set_actions(name, action)
