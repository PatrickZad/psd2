from .cuhk_sysu import load_cuhk_sysu
from .prw import load_prw
from .movie_net import load_movie_net

subset_names = ["Train", "Gallery"]


def load_cpm(dataset_dirs, subset="Train"):
    c_datas = load_cuhk_sysu(dataset_dirs[0], subset)
    p_datas = load_prw(dataset_dirs[1], subset)
    m_datas = load_movie_net(dataset_dirs[2], subset)
    return c_datas + p_datas + m_datas
