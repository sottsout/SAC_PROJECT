
from gymnasium import ObservationWrapper, spaces

# ------------------------------------------------------------------
# 1.  ΟΡΙΖΕΙΣ ΤΟ WRAPPER
class ObsOnlyWrapper(ObservationWrapper):
    """
    Επιστρέφει μόνο το obs["observation"] ώστε ο agent να λαμβάνει flat vector.
    """
    def __init__(self, env):
        super().__init__(env)
        # η νέα observation_space είναι ΜΟΝΟ το inner Box
        self.observation_space = env.observation_space["observation"]

    def observation(self, obs):
        # κρατάμε το flat vector και αγνοούμε goals
        return obs["observation"]

