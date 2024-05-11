import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import itertools


def inner(vec1, vec2):
    return np.dot(vec1.reshape(-1,).T, vec2.reshape(-1,))

def col_and_sgn(pauli_word, row_binary):
    col_binary = []
    s = 1.
    for j, p in enumerate(pauli_word):
        if p == 0:
            col_binary.append(row_binary[j])
        if p == 1:
            col_binary.append((row_binary[j]+1)%2)
        if p == 2:
            col_binary.append((row_binary[j]+1)%2)
            s *= -1j * (-1) ** row_binary[j]
        if p == 3:
            col_binary.append(row_binary[j])
            s *= (-1) ** row_binary[j]
    return tuple(col_binary), s

def gen_pauli_sparse(pauli_word):
    n = len(pauli_word)
    a = scipy.sparse.lil_matrix((2**n,2**n), dtype=complex)
    for row_binary in itertools.product([0,1], repeat=n):
        col_binary, s = col_and_sgn(pauli_word, row_binary)
        row = np.array(row_binary).dot(2 ** np.arange(len(row_binary))[::-1])
        col = np.array(col_binary).dot(2 ** np.arange(len(col_binary))[::-1])
        a[row, col] = s
    return a.tocsr()

def vec_to_sparse(vec):
    n = len(vec.shape)
    a = scipy.sparse.lil_matrix((2**n,2**n), dtype=complex).tocsr()
    for pauli_word in itertools.product([0,1,2,3], repeat=n):
        a += gen_pauli_sparse(pauli_word) * vec[pauli_word]
    return a

def sparse_to_vec(a, n):
    shape = tuple([4]*n)
    vec = np.zeros(shape)
    for pauli_word in itertools.product([0,1,2,3], repeat=n):
        p = gen_pauli_sparse(pauli_word)
        vec[pauli_word] = (a.conj().multiply(p)).sum().real / 2**n
    return vec

def gen_random_axis_angle(n):
    axan = np.zeros((n, 3))
    for k in range(n):
        #球面上ランダム回転軸とランダムな回転角を選ぶ
        u, v, angle = np.random.random_sample(3)
        vz = -2 * u + 1
        vx = np.sqrt(1-vz**2) * np.cos(2*np.pi*v)
        vy = np.sqrt(1-vz**2) * np.sin(2*np.pi*v)
        axan[k] = np.array([np.pi * (angle - 0.5) * vx,
                            np.pi * (angle - 0.5) * vy,
                            np.pi * (angle - 0.5) * vz])
    return axan

def axis_angle_to_vec(axan, n):
    shape = tuple([4]*n)
    vec = np.zeros(shape)
    for k in range(n):
        for xyz in [1,2,3]:
            ind = tuple([0] * k +[xyz] + [0] * (n - k - 1))
            vec[ind] = axan[k, xyz-1]
    return vec

def small_rotation_sandwich(vec1, vec2, eps, n, isvec1=True, isvec2=True):
    if isvec1:
        H1 = vec_to_sparse(vec1)
    else:
        H1 = vec1
    if isvec2:
        H2 = vec_to_sparse(vec2)
    else:
        H2 = vec2
    R = scipy.sparse.linalg.expm(H2 * -1j * eps)
    R_inv = scipy.sparse.linalg.expm(H2 * 1j * eps)
    return sparse_to_vec(R * H1 * R_inv, n)

def i_commutation(vec1, vec2, n, rot=False):
    a1 = vec_to_sparse(vec1)
    a2 = vec_to_sparse(vec2)
    a = 1.j * (a1 * a2 - a2 * a1)
    if rot:
        axan = gen_random_axis_angle(n)
        vec_rot = axis_angle_to_vec(axan, n)
        vec = small_rotation_sandwich(a, vec_rot, -1, n, isvec1=False)
        return vec, axan
    return sparse_to_vec(a, n)

def pauli_word_to_vec(pauli_word):
    n = len(pauli_word)
    shape = tuple([4] * n)
    vec = np.zeros(shape)
    vec[pauli_word] = 1
    return vec

def complicate_from_vec(vec, N, n, seed=None):
    vec_k = vec.copy()
    U_lis1 = []
    U_lis2 = []
    if seed:
        np.random.seed(seed)
    for k in range(N):
        rand_pauli_word = tuple(np.random.choice([0,1,2,3], n))
        theta = (np.random.rand() - 0.5) * np.pi
        vec_k = small_rotation_sandwich(vec_k, pauli_word_to_vec(rand_pauli_word), theta, n)
        U_lis1.append(rand_pauli_word)
        U_lis2.append(theta)
    return vec_k, U_lis1, U_lis2

def random_vec(n, seed=None):
    shape = tuple([4] * n)
    if seed:
        np.random.seed(seed)
    return np.random.random_sample(shape) / (4 ** n)




