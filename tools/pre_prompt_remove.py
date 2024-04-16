import sys
sys.path.append("./")
import pickle
from psd2.utils.file_io import PathManager
import torch
if __name__ == "__main__":
    filename = sys.argv[1]
    
    if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                print("Reading a file from '{}'".format(data["__author__"]))
                loaded=data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                loaded={
                    "model": data,
                    "__author__": "Caffe2",
                    "matching_heuristics": True,
                }
    elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            loaded={
                "model": model_state,
                "__author__": "pycls",
                "matching_heuristics": True,
            }

    loaded = torch.load(filename, map_location=torch.device("cpu"))  # load native pth checkpoint
    if "model" not in loaded:
            loaded = {"model": loaded}
    md=loaded["model"]
    for k,v in md.items():
         if k.startswith("stage_prompts"):
              num_t=v.shape[0]
              num_per_task=num_t//3
              v[0:num_per_task,:]=v[num_per_task:num_per_task*2,:]
              v[num_per_task:num_per_task*2,:]=v[num_per_task*2:num_per_task*3,:]
    torch.save(loaded,sys.argv[2])