import numpy as np
import matplotlib.pyplot as plt
from user_model import UserModel, Popularity

jaccard_list = list()

for _ in range(10):

    _jaccard_list = list()

    user_num = 100
    iter_num = 100

    user_model = UserModel(user_num, 10, iter=iter_num)

    neighborhood = user_model.neighborhood()

    popularity_model = Popularity(user_num, 10 * iter_num)

    ideal_idx = user_model.add_model()
    popularity_idx = user_model.add_model()

    for iter in range(iter_num):
        if iter < 10:
            user_model.recommend([[] for _ in range(user_num)], ideal_idx)
        else:
            user_model.recommend(user_model.get_ideal(ideal_idx), ideal_idx)
        
        ideal_jaccard = np.array([user_model.jaccard(ideal_idx, u, neighborhood[u]) for u in range(user_num)]).mean()

        if iter < 10:
            feedback = user_model.recommend([[] for _ in range(user_num)], popularity_idx)
            popularity_model.feedback(feedback)
        else:
            pred = popularity_model.predict(user_model.selected[popularity_idx])
            feedback = user_model.recommend(pred, popularity_idx)
            popularity_model.feedback(feedback)

        popularity_jaccard = np.array([user_model.jaccard(popularity_idx, u, neighborhood[u]) for u in range(user_num)]).mean()

        _jaccard_list.append(popularity_jaccard - ideal_jaccard)

        user_model.new_iter()

    jaccard_list.append(_jaccard_list)

jaccard_list = np.array(jaccard_list)

plt.clf()
plt.scatter(list(range(100)), jaccard_list.mean(axis=0))
plt.savefig('fig.png')