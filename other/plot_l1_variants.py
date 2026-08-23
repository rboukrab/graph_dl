import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# Grid of nonnegative magnitudes (penalties depend on |x|)
t = np.linspace(0, 5, 1001)  # |x| from 0 to 5

# l0 (approx): exact l0 is 0 at 0 and 1 elsewhere; approximate numerically with a tiny threshold
eps = 1e-6
l0 = (t > eps).astype(float)

# l1
l1 = t

# lp with p = 0.5 (representative nonconvex lp)
p = 0.5
lp = t**p

# SCAD penalty (Fan & Li, 2001)
def scad_penalty(t, lam=1.0, a=3.7):
    rho = np.zeros_like(t)
    # region 1: 0 <= t <= lambda
    mask1 = (t <= lam)
    rho[mask1] = lam * t[mask1]
    # region 2: lambda < t <= a*lambda
    mask2 = (t > lam) & (t <= a*lam)
    rho[mask2] = (-t[mask2]**2 + 2*a*lam*t[mask2] - lam**2) / (2*(a-1))
    # region 3: t > a*lambda
    mask3 = (t > a*lam)
    rho[mask3] = (lam**2 * (a+1)) / 2.0
    return rho

# MCP penalty (Zhang, 2010)
def mcp_penalty(t, lam=1.0, gamma=3.0):
    rho = np.zeros_like(t)
    mask1 = (t <= gamma * lam)
    rho[mask1] = lam * t[mask1] - (t[mask1]**2) / (2 * gamma)
    mask2 = (t > gamma * lam)
    rho[mask2] = 0.5 * gamma * lam**2
    return rho

lam = 1.0
scad = scad_penalty(t, lam=lam, a=3.7)
mcp = mcp_penalty(t, lam=lam, gamma=3.0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(t, l0, label=r'$\ell_0$ (approx)')
plt.plot(t, l1, label=r'$\ell_1$')
plt.plot(t, lp, label=r'$\ell_p,\ p={0.5}$')
plt.plot(t, scad, label='SCAD ($a=3.7, \\lambda=1$)')
plt.plot(t, mcp, label='MCP ($\\gamma=3, \\lambda=1$)')

plt.xlabel(r'$t = |x|$')
plt.ylabel('Penalty value')
plt.title('Comparison: $\\ell_0$, $\\ell_1$, $\\ell_p$, SCAD and MCP penalties')
plt.ylim([-0.1, 6])
plt.xlim([0, 5])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
