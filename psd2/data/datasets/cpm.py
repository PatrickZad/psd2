from .cuhk_sysu import load_cuhk_sysu
from .prw import load_prw
from .movie_net import load_movie_net

subset_names = ["Train"]


def load_cpm(dataset_dirs, subset="Train"):
    c_datas = load_cuhk_sysu(dataset_dirs[0], subset)
    id_offset=5532
    p_datas = load_prw(dataset_dirs[1], subset)
    for item in p_datas:
        for p_item in item["annotations"]:
            if p_item["person_id"]>=0:
                p_item["person_id"]+=id_offset
    id_offset=5532+483
    m_datas = load_movie_net(dataset_dirs[2], subset)
    for item in m_datas:
        for p_item in item["annotations"]:
            if p_item["person_id"]>=0:
                p_item["person_id"]+=id_offset
    return c_datas + p_datas + m_datas
