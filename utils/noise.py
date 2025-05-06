import numpy as np

class OUNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None, sigma_min=0.05, sigma_decay=0.99):
        self.mu = mu
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def decay_sigma(self):
        """Reduce sigma gradually to a minimum value."""
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        return self.__call__()
