import numpy as np
from scipy.spatial.distance import cdist
import math

class UserModel:

    def __init__(self, user_num, startup_iter, new_item_num_per_iter=10, iter=100, vec_dim=20):
        self.user_num = user_num
        self.startup_iter = startup_iter
        self.startup = True
        self.new_item_num_per_iter = new_item_num_per_iter
        self.item_num = new_item_num_per_iter * iter
        self.iter_num = iter
        mu_user = 10 * np.random.dirichlet(np.ones(vec_dim))
        mu_item = 0.1 * np.random.dirichlet(np.ones(vec_dim) * 100)
        self.user_vec = np.array([np.random.dirichlet(mu_user) for _ in range(user_num)])
        self.item_vec = np.array([np.random.dirichlet(mu_item) for _ in range(self.item_num)])

        def alt_beta(mean, std=1e-5):
            try:
                alpha = math.pow(mean, 2) * (((1 - mean) / math.pow(std, 2)) - (1 / mean))
                beta = alpha * ((1 / mean) - 1)
                return np.random.beta(alpha, beta)
            except:
                return mean

        _beta_func = np.vectorize(alt_beta)
        self.row_utility = _beta_func(np.matmul(self.user_vec, self.item_vec.T))
        proportion = _beta_func(0.98 * np.ones(self.row_utility.shape))
        self.utility = self.row_utility * proportion
        self.iter = -1
        self.new_item = []
        self.selected = np.zeros([1, self.user_num, self.item_num])
        self.model_num = 0
        self.new_iter()

    def new_iter(self):
        if self.iter < self.iter_num - 1:
            self.iter += 1
            self.new_item = np.array([i for i in range(self.iter * self.new_item_num_per_iter, (self.iter + 1) * self.new_item_num_per_iter)])
            np.random.shuffle(self.new_item)
            if self.iter >= self.startup_iter:
                self.startup = False

    def add_model(self):
        self.model_num += 1
        if self.model_num != 1:
            self.selected = np.concatenate((self.selected, np.zeros([1, self.user_num, self.item_num])), axis=0)
        return self.model_num - 1

    def interleave(self, rank_list):
        if len(rank_list) > 10:
            _tmp = rank_list[10:]
        else:
            _tmp = np.array([])
        interleaved = np.array([])
        for i in range(10):
            try:
                interleaved = np.append(interleaved, rank_list[i])
            except:
                pass
            interleaved = np.append(interleaved, self.new_item[i])
        return np.append(interleaved, _tmp)

    def recommend_oneuser(self, rank_list, u, model_idx):
        if self.startup:
            rank_list = self.new_item
        else:
            rank_list = np.array(rank_list)
            rank_list = self.interleave(rank_list)
            rank_list = rank_list.astype(int)
        select_item = -1
        max_score = 0
        for r, i in enumerate(rank_list):
            score = math.pow(r + 1, -0.8) * self.utility[u, i]
            if score > max_score and not self.selected[model_idx, u, i]:
                max_score = score
                select_item = i
        self.selected[model_idx, u, select_item] = 1
        return select_item

    def recommend(self, rank_list, model_idx):
        return [self.recommend_oneuser(r_list, u, model_idx) for u, r_list in enumerate(rank_list)]

    def get_ideal(self, idx):
        if self.startup:
            return np.argsort(self.row_utility[:, :self.iter * self.new_item_num_per_iter] * (1 - self.selected[idx, :, :self.iter * self.new_item_num_per_iter]))
        else:
            return np.argsort(self.row_utility[:, :self.startup_iter * self.new_item_num_per_iter] * (1 - self.selected[idx, :, :self.startup_iter * self.new_item_num_per_iter]))

    def jaccard(self, model_idx, u, v):
        d_u = np.where(self.selected[model_idx, u, :] == 1)[0]
        d_v = np.where(self.selected[model_idx, v, :] == 1)[0]
        return len(np.intersect1d(d_u, d_v)) / len(np.union1d(d_u, d_v))

    def neighborhood(self):
        _d = 1 - cdist(self.user_vec, self.user_vec, 'cosine')
        return (_d - np.identity(self.user_num)).argmax(axis=1)

class Popularity:

    def __init__(self, user_num, item_num):
        self.item_score = np.zeros(item_num)
        self.user_num = user_num

    def feedback(self, selected_list):
        selected_list = np.array(selected_list).flatten()
        for i in selected_list:
            self.item_score[i] += 1
    
    def predict(self, selected, top_k=50):
        item_score = np.tile(self.item_score, (self.user_num, 1))
        result = np.argsort(-1 * item_score * (1 - selected))
        if len(result) < top_k:
            return result
        return result[:, :top_k]


# print(pop_model.predict(10))

# # Sample

# ###### User Modeling Initialization ######
# startup_iter = 20
# user_num = 100
# test = UserModel(user_num, startup_iter)
# idx = test.add_model() # assume there's only one model now

# ###### start-up ######
# for t in range(startup_iter):
#     test.recommend([None]*user_num, idx) 
#     test.new_iter()

# ###### Get data & Train Model######
# ## Train
# collborative_user_item_table = test.selected[idx]
# mf = Matrix_Factorization(collborative_user_item_table)
# mf.fit()

# ###### User Feedback(Interaction) & Retrain The Model ######
# while True:
#     ## Inference
#     top_k = 30 # assume recommend 30 items to each user
#     rank_list = mf.recommend(top_k=30) # size of rank_list : (user_num, 30)
#     test.recommend(rank_list, idx) 
#     test.new_iter()
#     ## Retrain Model
#     mf.update_table(test.selected[idx])
#     mf.fit()
