
# ---------------------------------------------------------
# Import libraries
# ---------------------------------------------------------
import numpy as np
from numba import njit
import random
from scipy.stats import beta, binom
import time


#################
# First problem # 
#################

# ---------------------------------------------------------
# Required parameters
# ---------------------------------------------------------
T = 10          # number of days
x_max = 150     # maximum inventory level
x0 = 130        # initial inventory level
p = 18          # selling price
c = 10          # purchasing cost
h = 1           # overnight holding cost (storage fee)
u_max = 70      # maximum order quantity per day

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
np.random.seed(123)
random.seed(123)

# ---------------------------------------------------------
# Generate wages for the starting day τ
# ---------------------------------------------------------
def generate_w(tau):
    week = np.array([2, 2, 2, 2, 2, 4, 4], dtype=np.float64)
    w = np.zeros(T, dtype=np.float64)
    for i in range(T):
        w[i] = week[(tau + i) % 7]
    return w

# ---------------------------------------------------------
#  Generate weekly demand using a Beta distribution
# ---------------------------------------------------------
def weekly_demand(alpha_min, alpha_max, beta_min, beta_max, scale):
    alpha_param = round(random.uniform(alpha_min,alpha_max),2)
    beta_param = round(random.uniform(beta_min,beta_max),2)

    days = np.arange(7)
    d = (days + 0.5) / 7

    pdf_vals = beta.pdf(d, alpha_param, beta_param)
    pmf = pdf_vals / pdf_vals.sum()

    expected_demand = np.ceil(scale * pmf).astype(np.int64) # ceiling since demand must be an integer

    return days, expected_demand, alpha_param, beta_param

# ---------------------------------------------------------
# Dynamic programming (DP)
# ---------------------------------------------------------
@njit
def dp_kernel(V, v, T, p, c, h, x_max, u_max, wages, D):
  for i in range(T-1, -1, -1):
    for x in range(x_max + 1):
      for u in range(u_max + 1):

        S = min(D[i], x + u) # how much is actually sold

        x_next = int(min(max(x + u - S, 0), x_max))

        fpomocna = 0

        if x_next <= x_max:
            f0 = p*S - c*u - wages[i]*S - h * x_next
            fpomocna = f0 + V[i+1, x_next]

        if fpomocna > V[i, x]:
          V[i, x] = fpomocna
          v[i, x] = u


# ---------------------------------------------------------
# Solution
# ---------------------------------------------------------
def solve(tau): # tau is the given starting day

    wages = generate_w(tau)

    D = np.zeros(T, dtype=np.int64)
    for i in range(T):
      D[i] = int(D_week[(tau + i) % 7])

    print("Demand parameters:")
    print(f"alpha = {alpha_param}, beta = {beta_param}")
    print("Fixed demand over 10 days:", D)

    V = np.full((T+1, x_max+1), -np.inf, dtype=np.float64)
    v = np.zeros((T, x_max+1), dtype=np.int64)
    V[T, :] = 0.0

    dp_kernel(V, v, T, p, c, h, x_max, u_max, wages, D)

    return V, v

# ---------------------------------------------------------
# Choose the optimal starting day
# ---------------------------------------------------------

# weekday_names = ["Pondelok","Utorok","Streda","Štvrtok","Piatok","Sobota","Nedeľa"]
weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

best_tau = None
best_val = -1e300
best_V, best_v = None, None

days, D_week, alpha_param, beta_param = weekly_demand(alpha_min=3, alpha_max=6, beta_min=2, beta_max=4, scale = 150)

for tau in range(7):
    print("-------------------------------")
    print(f"Starting day τ = {tau} ({weekday_names[tau]}): ")
    V, v = solve(tau)
    val0 = V[0, x0]
    print(f"Total profit over 10 days = {val0:.2f} €")

    if val0 > best_val:
        best_val = val0
        best_tau = tau
        best_V = V
        best_v = v

print(f"\nOptimal starting day τ = {best_tau} ({weekday_names[best_tau]})")
print(f"otal profit for initial inventory  x0={x0}: {best_val:.2f} €\n")

# ---------------------------------------------------------
# Example of the optimal control (policy)
# ---------------------------------------------------------
x = x0
for t in range(T):
    weekday = weekday_names[(best_tau + t) % 7]
    u_opt = int(best_v[t,x])

    D_t = int(D_week[(best_tau + t) % 7])
    S = min(D_t, x + u_opt)

    x_next = int(min(max(x + u_opt - S, 0), x_max))

    print(
        f"t={t}, {weekday}: "
        f"x_start = {x}, "
        f"u* = {u_opt}, "
        f"S = {S:.2f}, "
        f"x_end = {x_next:.2f}")

    x = int(x_next)





#############
# Extension #
#############

# ---------------------------------------------------------
# Required parameters
# ---------------------------------------------------------

T = 10                 # number of days
x1_max = 30            # maximum inventory level for product 1
x2_max = 25            # maximum inventory level for product 2
u1_max = 25            # maximum order quantity for product 1 per day
u2_max = 20            # maximum order quantity for product 2 per day

c1, c2 = 10, 12        # purchasing costs for product 1 and product 2
p1, p2 = 18, 20        # selling prices for product 1 and product 2
h1, h2 = 2, 3          # overnight holding costs (storage fees) for product 1 and product 2

w = np.array([2,2,2,2,2,4,4,2,2,2])     # wages (sales start on Monday)

# ---------------------------------------------------------
# Binomial distribution for random demands
# ---------------------------------------------------------
prob1 = 0.7    # product 1
prob2 = 0.65   # product 2

z1_max = 50
z2_max = 45

states1 = np.arange(0, z1_max + 1)
states2 = np.arange(0, z2_max + 1)

probs1 = binom.pmf(states1, z1_max, prob1)
probs2 = binom.pmf(states2, z2_max, prob2)

# ---------------------------------------------------------
# Dynamic programming - 2 products (2 state variables, 2 controls, 2 random variables)
# ---------------------------------------------------------
@njit
def dp_kernel_2prod(V, v1, v2,
                    c1, p1, h1, c2, p2, h2,
                    probs1, probs2,
                    T, x1_max, x2_max, u1_max, u2_max,
                    w):

    for i in range(T-1, -1, -1):
        wage = w[i]
        for x1 in range(x1_max + 1):
            for x2 in range(x2_max + 1):
                for u1 in range(u1_max + 1):
                    for u2 in range(u2_max + 1):

                        fpomocna = 0.0

                        for z1 in range(len(probs1)):
                            for z2 in range(len(probs2)):
                                pz = probs1[z1] * probs2[z2]  # independent demands

                                s1 = min(x1 + u1, z1)
                                s2 = min(x2 + u2, z2)

                                x1_next = min(max(x1 + u1 - s1, 0), x1_max)
                                x2_next = min(max(x2 + u2 - s2, 0), x2_max)

                                f = (
                                    p1 * s1 - c1 * u1 - h1 * x1_next - wage * s1 +
                                    p2 * s2 - c2 * u2 - h2 * x2_next - wage * s2
                                )

                                fpomocna += pz * (f + V[i+1, x1_next, x2_next])

                        if fpomocna > V[i, x1, x2]:
                            V[i, x1, x2] = fpomocna
                            v1[i, x1, x2] = u1
                            v2[i, x1, x2] = u2

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
V = np.full((T+1, x1_max+1, x2_max+1), -np.inf)
v1 = np.full((T, x1_max+1, x2_max+1), -np.inf)
v2 = np.full((T, x1_max+1, x2_max+1), -np.inf)

for x1 in range(x1_max + 1):
    for x2 in range(x2_max + 1):
        V[T, x1, x2] = c1 * x1 + c2 * x2 

x1_0 = 5  # initial inventory of product 1
x2_0 = 5  # initial inventory of product 2

# ---------------------------------------------------------
# Computation
# ---------------------------------------------------------
start_time = time.time()

dp_kernel_2prod(
    V, v1, v2,
    c1, p1, h1, c2, p2, h2,
    probs1, probs2,
    T, x1_max, x2_max, u1_max, u2_max,
    w
)
end_time = time.time()

print("Maximum expected profit:", V[0, x1_0, x2_0])
print(f"Computation time: {end_time - start_time:.4f} sec")



