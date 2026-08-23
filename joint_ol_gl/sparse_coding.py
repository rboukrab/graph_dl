import torch
import numpy as np
from torch.nn.functional import softshrink

def _ISTA(x, D, alpha, tolerance, max_iter, device):

    batch_size = x.shape[0]
    n_atoms = D.shape[1]

    y = torch.zeros((batch_size, n_atoms), device=device)
    L = torch.abs(torch.linalg.eigvalsh(D.T @ D)).max()
    eta = 1 / L

    iter = 0
    error = tolerance + 1
    while error > tolerance and iter < max_iter:

        y_old = y.detach().clone()
        y = softshrink(y + eta * (x - y @ D.T) @ D, alpha)
        err = torch.linalg.norm(y - y_old) / batch_size
        if err < tolerance:
            break
        iter += 1
    return y


def _AIHT(x, A, m, M, thresh, device):
    """
    Accelerated iterative Hard thresholding algorithm that keeps exactly M elements
    in each iteration. This algorithm includes an additional double
    overrelaxation step that significantly improves convergence speed without
    destroiing any of the theoretical guarantees of the IHT algorithm
    detrived in [1], [2] and [3].

    This algorithm is used to solve the problem A*z=x

    Inputs:
     x: observation vector to be decomposed
     A: it can be a (nxm) matrix that gives the effect of the forward matrix A on a vector or an operator that does the same
     AT: it can be a (nxm) matrix that gives the effect of the backward matrix A.T on a vector or an operator that does the same
     m: length of the solution vector s
     M: number of non-zero elements to keep in each iteration
     thresh: stopping criterion

    Outputs:
     s: solution vector
     err_mse: vector containing mse of approximation error for each iteration
    """

    n = x.shape[1]

    sigsize = (x @ x.T).diag() / n
    oldERR = sigsize
    err_mse = []
    iter_time = []
    STOPTOL = 1e-16
    MAXITER = 100
    verbose = True
    initial_given = 0
    s_initial = torch.zeros((x.shape[0], m), device=device)
    MU = 0
    acceleration = 0
    Count = 0

    s_initial = torch.zeros((x.shape[0], m), device=device)
    Residual = x
    s = s_initial.detach().clone()
    Ps = torch.zeros((x.shape[0], n), device=device)
    oldErr = sigsize

    x_test = torch.randn((x.shape[0], m), device=device)
    x_test = x_test / torch.linalg.norm(x_test, dim=1).unsqueeze(1)
    nP = torch.linalg.norm(A @ x_test.T, dim=0)
    if (torch.abs(MU * nP) > 1).any():
        exit('WARNING! Algorithm likely to become unstable. Use smaller step-size or || P ||_2 < 1.')

    # Main algorithm
    t = 0
    done = False
    iteration = 1
    min_mu = torch.tensor(1e5)
    max_mu = torch.tensor(0)

    while (not done):
        Count += 1
        if (MU == 0):

            # Calculate optimal step size and do line search
            if ((Count > 1) & (acceleration == 0)):
                s_very_old = s_old
            s_old = s
            IND = s != 0
            d = (A.T @ Residual.T).T

            # If the current vector is zero, we take the largest element in d
            if (torch.sum(IND, dim=1) == 0).any():
                idx_sort = torch.where(torch.sum(IND, dim=1) == 0)[0]
                sortind = torch.argsort(torch.abs(d[idx_sort]), dim=1, descending=True)
                IND[idx_sort] = torch.zeros((idx_sort.shape[0], IND.shape[1]), dtype=torch.bool, device=device).scatter_(1, sortind[:, 0:M], 1)

            id = IND * d
            Pd = (A @ id.T).T
            mu = (id @ id.T).diag() / (Pd @ Pd.T).diag()
            max_mu = torch.maximum(mu, max_mu.clone().detach())
            min_mu = torch.minimum(mu, min_mu.clone().detach())
            mu = min_mu
            s = s_old + mu.unsqueeze(1) * d
            sortind = torch.argsort(torch.abs(s), dim=1, descending=True)
            # s[sortind[M:]] = 0
            s.scatter_(1, sortind[:, M:], 0)

            if ((Count > 1) & (acceleration == 0)):
                very_old_Ps = old_Ps
            old_Ps = Ps
            Ps = (A @ s.T).T
            Residual = x - Ps

            if ((Count > 2) & (acceleration == 0)):
                # First overrelaxation
                Dif = (Ps - old_Ps)
                a1 = (Dif @ Residual.T).diag() / (Dif @ Dif.T).diag()
                z1 = s + a1.unsqueeze(1) * (s - s_old)
                Pz1 = (1 + a1.unsqueeze(1)) * Ps - a1.unsqueeze(1) * old_Ps
                Residual_z1 = x - Pz1

                # Second overrelaxation
                Dif = Pz1 - very_old_Ps
                a2 = (Dif @ Residual_z1.T).diag() / (Dif @ Dif.T).diag()
                z2 = z1 + a2.unsqueeze(1) * (z1 - s_very_old)

                # Threshold z2
                sortind = torch.argsort(torch.abs(z2), dim=1, descending=True)
                z2.scatter_(1, sortind[:, M:], 0)
                Pz2 = (A @ z2.T).T
                Residual_z2 = x - Pz2

                # Decide if z2 is any good
                if ((Residual_z2 @ Residual_z2.T).diag() / (Residual @ Residual.T).diag() < 1).any():
                    idx_z2 = torch.where((Residual_z2 @ Residual_z2.T).diag() / (Residual @ Residual.T).diag() < 1)[0]
                    s[idx_z2] = z2[idx_z2]
                    Residual[idx_z2] = Residual_z2[idx_z2]
                    Ps[idx_z2] = Pz2[idx_z2]

            # if (acceleration > 0):
            # s, Residual = mySubsetCG(x, s, P, Pt

            # Calculate step-size requirements
            omega = (torch.linalg.norm(s - s_old) / torch.linalg.norm(Ps - old_Ps)) ** 2

            # As long as the support changes and mu > omega, we decrease mu
            while ((mu > 1.5 * omega).any() & (torch.sum(torch.logical_xor(IND, s != 0)) != 0) & (torch.sum(IND) != 0)):
                print
                "Decreasing mu"

                # We use a simple line search, halving mu in each step
                mu = mu / 2
                s = s_old + mu.unsqueeze(1) * d
                sortind = torch.argsort(torch.abs(s), dim=1, descending=True)
                s.scatter_(1, sortind[:, M:], 0)
                Ps = (A @ s.T).T

                # Calculate optimal step size and do line search
                Residual = x - Ps
                if ((Count > 2) & (acceleration == 0)):

                    # First overrelaxation
                    Dif = (Ps - old_Ps)
                    a1 = (Dif @ Residual.T).diag() / (Dif @ Dif.T).diag()
                    z1 = s + a1.unsqueeze(1) * (s - s_old)
                    Pz1 = (1 + a1).unsqueeze(1) * Ps - a1.unsqueeze(1) * old_Ps
                    Residual_z1 = x - Pz1

                    # Second overrelaxation
                    Dif = Pz1 - very_old_Ps
                    a2 = (Dif @ Residual_z1.T).diag() / (Dif @ Dif.T).diag()
                    z2 = z1 + a2.unsqueeze(1) * (z1 - s_very_old)

                    # Threshold z2
                    sortind = torch.argsort(torch.abs(z2), dim=1, descending=True)
                    z2.scatter_(1, sortind[:, M:], 0)
                    Pz2 = (A @ z2.T).T
                    Residual_z2 = x - Pz2

                    # Decide if z2 is any good
                    if ((Residual_z2 @ Residual_z2.T).diag() / (Residual @ Residual.T).diag() < 1).any():
                        idx_z2 = torch.where((Residual_z2 @ Residual_z2.T).diag() / (Residual @ Residual.T).diag() < 1)[0]
                        s[idx_z2] = z2[idx_z2]
                        Residual[idx_z2] = Residual_z2[idx_z2]
                        Ps[idx_z2] = Pz2[idx_z2]

                # Calculate step-size requirements
                omega = (torch.linalg.norm(s - s_old) / torch.linalg.norm(Ps - old_Ps)) ** 2

            ERR = (Residual @ Residual.T).diag() / n
            err_mse.append(ERR)

            # Are we done yet?
            gap = torch.linalg.norm(s - s_old, dim=1) ** 2 / m
            if (gap < thresh).all() or iteration == MAXITER:
                done = True

            if (not done):
                iteration += 1
                oldERR = ERR
            if (verbose):
                print
                "Iter={0} - gap={1} - target={2}".format(Count, gap, thresh)
    return s, err_mse
