import numpy as np

def generate_points(n, d, distribution='uniform', seed=None):
    """
    Generate n random points in R^d for smallest enclosing ball problem.
    
    Parameters:
    -----------
    n : int
        Number of points
    d : int
        Dimension of space
    distribution : str
        'uniform': uniform in [-1, 1]^d
        'normal': Gaussian standard
        'sphere': uniformly on unit sphere surface
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    points : ndarray, shape (d, n)
        Matrix where each column is a point
    """
    if seed is not None:
        np.random.seed(seed)
    
    if distribution == 'uniform':
        # Uniform distribution in [-1, 1]^d
        points = np.random.uniform(-1, 1, (d, n))
        
    elif distribution == 'normal':
        # Gaussian distribution N(0, I)
        points = np.random.randn(d, n)
        
    elif distribution == 'sphere':
        # Uniformly on unit sphere surface
        points = np.random.randn(d, n)
        # Normalize each column to unit length
        points = points / np.linalg.norm(points, axis=0, keepdims=True)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Validation: check for duplicate points (unlikely but possible)
    unique_points = np.unique(points, axis=1)
    if unique_points.shape[1] < n:
        print(f"Warning: {n - unique_points.shape[1]} duplicate points generated")
    
    # Validation: check rank (points should span d-dimensional space ideally)
    if n >= d:
        rank = np.linalg.matrix_rank(points)
        if rank < min(d, n):
            print(f"Warning: Points have rank {rank} < min(d={d}, n={n})")
    
    return points


def points_to_qp_format(points):
    """
    Convert point matrix to QP format for smallest enclosing ball.
    
    From Schönherr formulation:
        min x^T C^T C x - sum_i ||p_i||^2 x_i
        
    This expands to:
        min (1/2) x^T (2 C^T C) x + (-||p_i||^2)^T x
        
    But we need Q to be positive semidefinite for convex QP.
    C^T C is always PSD (symmetric, eigenvalues >= 0).
    
    Parameters:
    -----------
    points : ndarray, shape (d, n)
        Matrix of n points in R^d
        
    Returns:
    --------
    Q : ndarray, shape (n, n)
        Quadratic term matrix (Q = 2 * C^T * C, PSD)
    q : ndarray, shape (n,)
        Linear term vector (q_i = -||p_i||^2)
    A_eq : ndarray, shape (1, n)
        Equality constraint matrix (sum x_i = 1)
    b_eq : float
        Equality constraint RHS (= 1)
    """
    d, n = points.shape
    
    # Compute C^T C (Gram matrix - always PSD)
    CTC = points.T @ points  # n x n matrix
    
    # Verify it's symmetric
    if not np.allclose(CTC, CTC.T):
        CTC = (CTC + CTC.T) / 2  # Symmetrize
    
    # Quadratic term: Q = 2 * C^T * C (still PSD)
    Q = 2 * CTC
    
    # Verify positive semi-definiteness
    eigvals = np.linalg.eigvalsh(Q)
    min_eigval = np.min(eigvals)
    if min_eigval < -1e-10:
        print(f"WARNING: Q has negative eigenvalue: {min_eigval}")
    
    # Linear term: q_i = -||p_i||^2
    q = -np.sum(points**2, axis=0)  # n-vector
    
    # Equality constraint: sum(x_i) = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    
    return Q, q, A_eq, b_eq


def extract_solution(x_opt, points, obj_value):
    """
    Extract center and radius from optimal QP solution.
    
    Parameters:
    -----------
    x_opt : ndarray, shape (n,)
        Optimal weights
    points : ndarray, shape (d, n)
        Original points
    obj_value : float
        Optimal objective value
        
    Returns:
    --------
    center : ndarray, shape (d,)
        Center of smallest enclosing ball
    radius : float
        Radius of smallest enclosing ball
    active_points : ndarray
        Indices of points on the boundary (x_i > threshold)
    """
    # Center is weighted combination of points
    center = points @ x_opt  # d-vector
    
    # Radius² = -objective_value
    radius_squared = -obj_value
    radius = np.sqrt(max(0, radius_squared))  # Protect against numerical errors
    
    # Find active points (those with significant weight)
    threshold = 1e-6
    active_points = np.where(x_opt > threshold)[0]
    
    return center, radius, active_points


if __name__ == "__main__":
    # Test the data generation
    print("Testing data generation...")
    
    # Generate test data
    n, d = 50, 3
    points = generate_points(n, d, distribution='uniform', seed=42)
    print(f"\nGenerated {n} points in R^{d}")
    print(f"Points shape: {points.shape}")
    print(f"First point: {points[:, 0]}")
    
    # Convert to QP format
    Q, q, A_eq, b_eq = points_to_qp_format(points)
    print(f"\nQP format:")
    print(f"Q shape: {Q.shape}")
    print(f"q shape: {q.shape}")
    print(f"A_eq shape: {A_eq.shape}")
    print(f"b_eq: {b_eq}")
    
    print("\nData generation successful!")
