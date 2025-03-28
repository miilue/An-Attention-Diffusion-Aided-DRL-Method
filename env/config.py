import numpy as np
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# For service provider
NUM_EDGE_SERVER = 10  # number of edge servers
NUM_SERVICE_PROVIDERS = 4  # number of AIGC models deployed on each edge server
TOTAL_T_RANGE = np.arange(600, 1500, step=100)  # range of total t for service providers
# NUM_CPUS = 32  # number of logical cpu cores available
# NUM_GPUS = 8  # number of graphic cards available
# CPU_MEM = 128 * 2 ** 30  # total cpu memory available
# GPU_MEM = 24 * 2 ** 30  # gpu memory for each graphic card


# For user
NUM_USERS = 1000  # number of users to serve

# For user & service provider
# LOCATION_RANGE = [(0, 0), (100, 100)]  # [(x_min, y_min), (x_max, y_max)]

# For task
NUM_TASK_TYPES = 4  # number of task types available
T_RANGE = np.arange(100, 250, step=10)  # range of t_T for diffusion algorithm
# Runtime for each image. The value is proportional to t_T in the diffusion algorithm.
RUNTIME = lambda t: (0.001 * t ** 2 + 2.5 * t - 14) * 60
# IMG_CHW = (3, 218, 178)  # (n_channel, height, width) generated image
# IMG_BUFFER = 8 * 2 ** 10  # 8KBytes per generated image, JPEG format, for storage and transmission
# CPU_MEM_OCCUPY = 2000 * 2 ** 20  # 4980MB CPU memory occupation per image and per run
# GPU_MEM_OCCUPY = 4000 * 2 ** 20  # 7468MB GPU memory occupation per image and per run

# For task generator
TOTAL_TIME = 1000000  # time duration of an episode
LAMBDA = 0.0015  # λ for Poisson distribution

# Reward function for an inpainted image. The value is related to t_T in the diffusion algorithm.
BETA = 1
TYPE1_RANGE = np.arange(0, 1, step=0.1)
TYPE2_RANGE = np.arange(0, 1, step=0.1)
TYPE3_RANGE = np.arange(0, 1, step=0.1)
TYPE4_RANGE = np.arange(0, 1, step=0.1)
AX_RANGE = np.arange(0, 100, step=10)
AY_RANGE = np.arange(0., 0.5, step=0.05)
BX_RANGE = np.arange(150, 250, step=10)
BY_RANGE = np.arange(0.5, 1., step=0.05)
REWARD = lambda ax, ay, bx, by, t: \
    (by - ay) / (bx - ax) * (t - ax) if ax <= t <= bx else (0 if t < ax else by - ay)
CRASH_PENALTY_COEF = 2.  # The penalty unit value for crash

