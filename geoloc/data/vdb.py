import os
import torch
import faiss
from tqdm import tqdm
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from ..config_parser import class_from_config
from ..utils import DEBUG


class VectorDatabase:

    def __init__(self, vdim: int, distfn: str = "l2", norm_vec: bool = True):
        self.vdim = vdim
        self.distfn = distfn
        self.norm_vec = norm_vec

        if self.distfn == "cosine":
            self.faiss_index = faiss.IndexFlatIP(vdim)
        elif self.distfn == "l2":
            self.faiss_index = faiss.IndexFlatL2(vdim)
        else:
            raise ValueError(f"Invalid distance function: {self.distfn}")

    def add(self, vectors: torch.Tensor):
        if self.norm_vec:
            vectors = F.normalize(vectors)
        self.faiss_index.add(vectors.cpu().numpy())

    def search(self, qu: torch.Tensor, k: int):
        if self.norm_vec:
            qu = F.normalize(qu)
        distances, indices = self.faiss_index.search(qu.cpu().numpy(), k)
        return distances, indices

    def size(self):
        return self.faiss_index.ntotal

    def save(self, savedir: str):
        index_savepath = os.path.join(savedir, "faiss_index.index")
        faiss.write_index(self.faiss_index, index_savepath)

    def _flush_buffer(self, buffer):
        if buffer:
            x_buffer = torch.vstack(buffer)
            self.add(x_buffer)
            buffer.clear()

    @torch.no_grad()
    def build(
        self,
        ref_image_dataloader,
        model,
        rotation_angles=None,
        device="cpu",
        buffer_size=6000,
    ):
        if rotation_angles is None:
            rotation_angles = [0]

        buffer = []
        for theta in rotation_angles:
            for i, ref in enumerate(tqdm(ref_image_dataloader)):
                if DEBUG > 1 and i > 10:
                    break
                image = ref["image"].to(device)
                image = TF.rotate(image, theta)
                x = model(image)
                if self.norm_vec:
                    x = F.normalize(x)

                buffer.append(x)
                if len(buffer) >= buffer_size:
                    self._flush_buffer(buffer)
        self._flush_buffer(buffer)


def load_database(loaddir: str, config):
    """Load a faiss index from a file."""
    db_index_path = os.path.join(loaddir, "faiss_index.index")
    assert os.path.exists(db_index_path), f"Index file not found: {db_index_path}"
    db = class_from_config(config.vdb)
    db.faiss_index = faiss.read_index(db_index_path)
    return db
