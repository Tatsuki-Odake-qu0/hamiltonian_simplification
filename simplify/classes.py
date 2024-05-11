from .funcs import *
import matplotlib.pyplot as plt
import time

class simplifier:
    def __init__(self, f_v, tilde_f_v):
        self.f_v = f_v
        self.tilde_f_v = tilde_f_v

    def F(self, vec):
        vec2 = np.zeros_like(vec)
        n = len(vec.shape)
        for pauli_word in itertools.product([0,1,2,3], repeat=n):
            vec2[pauli_word] = self.f_v(pauli_word, vec[pauli_word])
        return vec2

    def tilde_F(self, vec):
        vec2 = np.zeros_like(vec)
        n = len(vec.shape)
        for pauli_word in itertools.product([0,1,2,3], repeat=n):
            vec2[pauli_word] = self.tilde_f_v(pauli_word, vec[pauli_word])
        return vec2

    def cost(self, vec):
        return inner(self.F(vec), vec)

    def cost_vs_angle(self, vec1, vec2, eps, n):
        return self.cost(small_rotation_sandwich(vec1, vec2, eps, n))

    def convergence(self, vec_ini, vec_current, n):
        vec_ini_1 = self.F(vec_ini) + self.tilde_F(vec_ini)
        vec_current_1 = self.F(vec_current) + self.tilde_F(vec_current)
        vec_e_ini = -1 * i_commutation(vec_ini_1, vec_ini, n)
        vec_e_current = -1 * i_commutation(vec_current_1, vec_current, n)
        return np.abs(vec_e_current.reshape(-1,)).sum() / np.abs(vec_e_ini.reshape(-1,)).sum()

    def __call__(self, vec, thres=0.01, n_wait=10, verbose=True, random=True, seed=None):
        start = time.time()
        n = len(vec.shape)
        n_wait = min(n_wait, 4**n)
        vec_k = vec.copy()
        U_lis1 = []
        U_lis2 = []
        cost_history = [self.cost(vec_k)]
        ind = 0
        ind_last = 0

        if random:
            if seed:
                np.random.seed(seed)
            while ind_last + n_wait >= ind:
                vec1 = self.F(vec_k) + self.tilde_F(vec_k)
                vec_e, axan = i_commutation(vec1, vec_k, n, rot=True)
                vec_frame = axis_angle_to_vec(axan, n)
                vec_e *= -1.
                i_e = np.abs(vec_e.reshape(-1,)).argmax().astype(int)
                pauli_word_e = tuple([int(s) for s in np.base_repr(i_e, 4).zfill(n)])
                vec_rot = np.sign(vec_e[pauli_word_e]) * pauli_word_to_vec(pauli_word_e)
                vec_rot = small_rotation_sandwich(vec_rot, vec_frame, 1, n)
                f = lambda eps: self.cost_vs_angle(vec_k, vec_rot, eps[0], n)
                res = scipy.optimize.minimize(fun = f,
                                            x0 = np.array([0]),
                                            bounds = ((-np.pi /2, np.pi / 2),))
                vec_k = small_rotation_sandwich(vec_k, vec_rot, res.x[0], n)
                cost_k = self.cost(vec_k)
                U_lis1.append([pauli_word_e, axan])
                U_lis2.append(res.x[0] * np.sign(vec_e[pauli_word_e]))
                cost_history.append(cost_k)

                if abs(res.x[0]) > thres:
                    ind_last = ind
                if verbose:
                    print(f'iteration:{ind}, ind_last:{ind_last}, axis:{[pauli_word_e, axan]}, angle:{res.x[0]  * np.sign(vec_e[pauli_word_e])}, cost:{cost_k}')
                ind += 1
        else:
            while ind_last + n_wait >= ind:
                vec1 = self.F(vec_k) + self.tilde_F(vec_k)
                vec_e = -1 * i_commutation(vec1, vec_k, n, rot=False)
                i_e = np.abs(vec_e.reshape(-1,)).argmax().astype(int)
                pauli_word_e = tuple([int(s) for s in np.base_repr(i_e, 4).zfill(n)])
                vec_rot = np.sign(vec_e[pauli_word_e]) * pauli_word_to_vec(pauli_word_e)
                f = lambda eps: self.cost_vs_angle(vec_k, vec_rot, eps[0], n)
                res = scipy.optimize.minimize(fun = f,
                                            x0 = np.array([0]),
                                            bounds = ((-np.pi /2, np.pi / 2),))
                vec_k = small_rotation_sandwich(vec_k, vec_rot, res.x[0], n)
                cost_k = self.cost(vec_k)
                U_lis1.append(pauli_word_e)
                U_lis2.append(res.x[0] * np.sign(vec_e[pauli_word_e]))
                cost_history.append(cost_k)

                if abs(res.x[0]) > thres:
                    ind_last = ind
                if verbose:
                    print(f'iteration:{ind}, ind_last:{ind_last}, pauli word:{pauli_word_e}, angle:{res.x[0] * np.sign(vec_e[pauli_word_e])}, cost:{cost_k}')
                ind += 1

        end = time.time()
        print("\nSimplification stopped")
        result = output(self.f_v, self.tilde_f_v, vec.copy(), n, thres, n_wait, random, seed, vec_k, U_lis1, U_lis2, ind, end-start, cost_history)
        return result

class output(simplifier):
    def __init__(self, f_v, tilde_f_v, vec_ini, n, thres, n_wait, random, seed, vec_fin, U_lis1, U_lis2, itera, comp_time, cost_history):
        super().__init__(f_v, tilde_f_v)
        self.vec_ini = vec_ini
        self.n = n
        self.thres = thres
        self.n_wait = n_wait
        self.random = random
        self.seed = seed
        self.vec_fin = vec_fin
        self.U_lis1 = U_lis1
        self.U_lis2 = U_lis2
        self.itera = itera
        self.comp_time = comp_time
        self.cost_history = cost_history
        self.cost_decrease = self.cost(vec_fin) / self.cost(vec_ini)
        self.relative_convergence = self.convergence(vec_ini, vec_fin, n)

    def simp_input(self, out="no_vec"):
        if out == "vec":
            print(f"vec:{self.vec_ini},\nn:{self.n},\nthres:{self.thres},\nn_wait:{self.n_wait},\nrandom:{self.random},\nseed:{self.seed}")
        if out == "no_vec":
            print(f"n:{self.n},\nthres:{self.thres},\nn_wait:{self.n_wait},\nrandom:{self.random}\nseed:{self.seed}")
        if out == "as_str":
            return f"n:{self.n}, thres:{self.thres}, n_wait:{self.n_wait}, random:{self.random}, seed:{self.seed}"

    def simp_change(self):
        print(f"The cost became {self.cost_decrease} times, and the convergence become {self.relative_convergence} times.")

    def cost_plot(self):
        s = self.simp_input(out="as_str")
        plt.plot(self.cost_history / self.cost_history[0])
        plt.xlabel("Iteration")
        plt.ylabel("Relative cost")
        plt.title("Cost history for "+s)
        plt.show()

    def histogram(self):
        weight_sum_ini = np.zeros(self.n+1)
        weight_sum_fin = np.zeros(self.n+1)
        for pauli_word in itertools.product([0,1,2,3], repeat=self.n):
            pauli_arr = np.array(pauli_word)
            i_w = (pauli_arr != 0).sum().astype(int)
            weight_sum_ini[i_w] += abs(self.vec_ini[pauli_word])
            weight_sum_fin[i_w] += abs(self.vec_fin[pauli_word])
        label = np.arange(self.n+1)
        p1 = plt.bar(label-0.1, weight_sum_ini, label=label, width=0.2, color="r")
        p2 = plt.bar(label+0.1, weight_sum_fin, label=label, width=0.2, color="b")
        plt.legend((p1[0], p2[1]), ("before", "after"))
        plt.title("abs sum of coefficients for different interation numbers")
        plt.xlabel("interation numbers (# non-identity)")
        plt.ylabel("abs sum of coefficients")
        plt.show()

    def __call__(self, vec):
        pass
