import sys

sys.path.append("./")
import fire
import torch
import torch.nn.functional as tF
from sklearn.cluster import KMeans

def main(img_feats,n_clusters):
    domain_feats = torch.load(img_feats)
    
    domain_feats = tF.normalize(domain_feats, dim=-1).cpu().numpy()
    clusters = (
                KMeans(n_clusters=n_clusters, random_state=0).fit(domain_feats).cluster_centers_
            )

    clusters=torch.tensor(clusters)
    print(clusters.shape)
    torch.save(clusters,img_feats[:-4]+"_{}centroids.pth".format(n_clusters))


if __name__ == "__main__":
    fire.Fire(main)
