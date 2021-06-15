import numpy as np
def compute_genre_distr(item_list):
    """
    Input : A list ,  list of items, item with item_id, genre and genre_scale(score)  
    Output : A dict
    """
    distr = {}
    for item in item_list:
        for genre, score in item.genres.items():
            genre_score = distr.get(genre, 0.)
            distr[genre] = genre_score + score

    # we normalize the summed up probability so it sums up to 1
    # and round it to three decimal places, adding more precision
    # doesn't add much value and clutters the output
    for item, genre_score in distr.items():
        normed_genre_score = round(genre_score / len(item_list), 3)
        distr[item] = normed_genre_score

    return distr
def distr_comparison_plot(interacted_distr, reco_distr, width=0.3):
    """
    Input : two dict
    eample for single dict:
        {'Comedy': 0.062,
         'Horror': 0.225,
         'Thriller': 0.163,
         'Action': 0.087,
         'Fantasy': 0.129,
         'Adventure': 0.108,
         'Drama': 0.029,
         'Sci-Fi': 0.071,
         'Mystery': 0.046,
         'Animation': 0.037,
         'Children': 0.029,
         'Romance': 0.013}
    """
    # the value will automatically be converted to a column with the
    # column name of '0'
    interacted = pd.DataFrame.from_dict(interacted_distr, orient='index')
    reco = pd.DataFrame.from_dict(reco_distr, orient='index')
    df = interacted.join(reco, how='outer', lsuffix='_interacted')

    n = df.shape[0]
    index = np.arange(n)
    plt.barh(index, df['0_interacted'], height=width, label='interacted distr')
    plt.barh(index + width, df['0'], height=width, label='reco distr')
    plt.yticks(index, df.index)
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.title('Genre Distribution between User Historical Interaction v.s. Recommendation')
    plt.ylabel('Genre')
    plt.show()
def compute_kl_divergence(interacted_distr, reco_distr, alpha=0.01):
    """
    Input : two dict
    example:
        {'Comedy': 0.062,
         'Horror': 0.225,
         'Thriller': 0.163,
         'Action': 0.087,
         'Fantasy': 0.129,
         'Adventure': 0.108,
         'Drama': 0.029,
         'Sci-Fi': 0.071,
         'Mystery': 0.046,
         'Animation': 0.037,
         'Children': 0.029,
         'Romance': 0.013}
    """
    
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    kl_div = 0.
    for genre, score in interacted_distr.items():
        reco_score = reco_distr.get(genre, 0.)
        reco_score = (1 - alpha) * reco_score + alpha * score
        kl_div += score * np.log2(score / reco_score)

    return kl_div

def compute_utility(reco_items, interacted_distr, lmbda=0.5):
    """
    Input : reco_items : a list of Item Class
            interacted_distr : compute_genre_distr(ideal_item_list)
    """
    
    """
    Our objective function for computing the utility score for
    the list of recommended items.

    lmbda : float, 0.0 ~ 1.0, default 0.5
        Lambda term controls the score and calibration tradeoff,
        the higher the lambda the higher the resulting recommendation
        will be calibrated. Lambda is keyword in Python, so it's
        lmbda instead ^^
    """
    reco_distr = compute_genre_distr(reco_items)
    kl_div = compute_kl_divergence(interacted_distr, reco_distr)

    total_score = 0.0
    for item in reco_items:
        total_score += item.score
    
    # kl divergence is the lower the better, while score is
    # the higher the better so remember to negate it in the calculation
    utility = (1 - lmbda) * total_score - lmbda * kl_div
    return utility
def calib_recommend(items, interacted_distr, topn, lmbda=0.5):
    """
    start with an empty recommendation list,
    loop over the topn cardinality, during each iteration
    update the list with the item that maximizes the utility function.
    """
    calib_reco = []
    for _ in range(topn):
        max_utility = -np.inf
        for item in items:
            if item in calib_reco:
                continue

            utility = compute_utility(calib_reco + [item], interacted_distr, lmbda)
            if utility > max_utility:
                max_utility = utility
                best_item = item

        calib_reco.append(best_item)
        
    return calib_reco
class Item:
    """
    Data holder for our item.
    
    Parameters
    ----------
    id : int
  
    genre : dict[str, float]
        The item/movie's genre distribution, where the key
        represents the genre and value corresponds to the
        ratio of that genre.

    score : float
        Score for the item, potentially generated by some
        recommendation algorithm.
    """
    def __init__(self, _id, genres, score=None):
        self.id = _id
        self.score = score
        self.genres = genres

def calib_recommend(items, interacted_distr, topn, lmbda=0.5):
    """
    start with an empty recommendation list,
    loop over the topn cardinality, during each iteration
    update the list with the item that maximizes the utility function.
    """
    calib_reco = []
    for _ in range(topn):
        max_utility = -np.inf
        for item in items:
            if item in calib_reco:
                continue

            utility = compute_utility(calib_reco + [item], interacted_distr, lmbda)
            if utility > max_utility:
                max_utility = utility
                best_item = item
        calib_reco.append(best_item)
        
    return calib_reco

ranking_list = [Item(0,genres={"A":1.0}, score=1.33),Item(7,genres={"B":1.0}, score=1.2)]
ideal_item_list = [Item(1,genres={"C":1.0},score=None), Item(2,genres={"B":1.0},score=None)]
ideal_item_distribution = compute_genre_distr(ideal_item_list)
#compute_utility(ranking_list, ideal_item_distribution)
calibrated_list = calib_recommend(ranking_list, ideal_item_distribution, 1)







