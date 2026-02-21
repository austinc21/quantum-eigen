import numpy as np
from scipy.linalg import eigh
import argparse
def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters
    ----------
    N : int
    Number of points in each dimension (N^2 total points).
    potential : str
    Choose the potential. 'well' or 'harmonic' examples.
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
    The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N) # grid spacing, can be arbitrary
    inv_dx2 = float(N * N) # 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)

    # Helper function to map (i,j) -> linear index
    def idx(i, j):
        return i * N + j
    
    # Potential function
    def V(i, j):
    # Example 1: infinite square well -> zero in interior, large outside
        if potential == 'well':
    # No boundary enforcement here, but can skip boundary wavefunction
            return 0.
    
    # Example 2: 2D harmonic oscillator around center
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            # Quadratic potential V = k * (x^2 + y^2)
            return 4. * (x**2 + y**2)
        else:
            return 0.
        
    # Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i,j)
            # Potential
            H[row, row] = -4. * inv_dx2 + V(i,j) # "Kinetic" ~ -4/dx^2 in 2D FD
            # Neighbors (assuming no boundary conditions or Dirichlet)
            if i > 0: # up
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1: # down
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0: # left
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1: # right
                H[row, idx(i, j+1)] = inv_dx2
    return H
    
def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
    Parameters
    ----------
    N : int
    Grid points in each dimension.
    potential : str
    Potential type.
    n_eigs : int
    Number of eigenvalues to return.
    Returns
    -------
    vals : array_like
    The lowest n_eigs eigenvalues sorted ascending.
    vecs : array_like
    The corresponding eigenvectors.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if potential not in ("well", "harmonic"):
        raise ValueError("potential must be 'well' or 'harmonic'.")

    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer.")
        if n_eigs > N * N:
            raise ValueError("n_eigs must be <= N^2.")
    H = build_2d_hamiltonian(N, potential)


    # Solve entire spectrum 
    vals, vecs = eigh(H)

# Save ground-state probability density 

    psi0 = vecs[:, 0]                    # lowest eigenvector
    psi0_grid = psi0.reshape(args.N, args.N)

    prob_density = np.abs(psi0_grid)**2  # |psi|^2

    np.savetxt(f"psi0_N{args.N}.txt", prob_density)


    # Sort
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Hamiltonian eigen solver")

    parser.add_argument("--N", type=int, default=10,
                        help="Grid size (NxN)")

    parser.add_argument("--potential", type=str, default="well",
                        choices=["well", "harmonic"],
                        help="Potential type")

    parser.add_argument("--n_eigs", type=int, default=5,
                        help="Number of eigenvalues to print")

    args = parser.parse_args()

    vals, vecs = solve_eigen(args.N, args.potential, args.n_eigs)

    print(f"Lowest {args.n_eigs} eigenvalues:", vals)
    np.savetxt(f"eigs_N{args.N}.txt", vals)

