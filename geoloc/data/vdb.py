import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torchvision.transforms.functional as TF

try:
    from asmk import asmk_method
    from asmk import io_helpers
except ImportError:
    asmk_method = None
    io_helpers = None
    print("Warning: asmk package not found. Mast3rASMKVectorDatabase will not work.")

from ..config_parser import class_from_config
from ..utils import DEBUG


class VectorDatabase:

    def __init__(self, vdim: int, distfn: str = "l2", norm_vec: bool = True, faiss_gpu: bool = False):
        self.vdim = vdim
        self.distfn = distfn
        self.norm_vec = norm_vec
        self.faiss_gpu = faiss_gpu
        self.vdbdir = None
        if self.distfn == "cosine":
            self.faiss_index = faiss.IndexFlatIP(vdim)
        elif self.distfn == "l2":
            self.faiss_index = faiss.IndexFlatL2(vdim)
        else:
            raise ValueError(f"Invalid distance function: {self.distfn}")
        
        if self.faiss_gpu:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        self.theta_buffer = []

    def add(self, vectors: torch.Tensor):
        if self.norm_vec:
            vectors = F.normalize(vectors)
        self.faiss_index.add(np.ascontiguousarray(vectors.cpu().numpy()))

    def search(self, qu: torch.Tensor, k: int):
        if self.norm_vec:
            qu = F.normalize(qu)
        distances, indices = self.faiss_index.search(qu.cpu().numpy(), k)
        return distances, indices

    def size(self):
        return self.faiss_index.ntotal

    def save(self, savedir: str):
        self.vdbdir = savedir
        index_savepath = os.path.join(savedir, "faiss_index.index")
        if self.faiss_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, index_savepath)
        else:
            faiss.write_index(self.faiss_index, index_savepath)

        if len(self.theta_buffer) > 0:
            theta_savepath = os.path.join(savedir, "thetas.npy")
            np.save(theta_savepath, np.array(self.theta_buffer))

    def _flush_buffer(self, buffer):
        if buffer:
            x_buffer = torch.vstack(buffer)
            self.add(x_buffer)
            buffer.clear()

    def save_ref_salad_matrix(self, savedir: str, filename: str, salad_matrix: torch.Tensor):
        os.makedirs(savedir, exist_ok=True)
        salad_path = os.path.join(savedir, filename)
        np.save(salad_path, salad_matrix.cpu().numpy())

    @torch.no_grad()
    def build(
        self,
        ref_image_dataloader,
        model,
        rotation_angles=None,
        device="cpu",
        buffer_size=6000,
        verbose=True,
        save_salad_matrix=False,
        salad_matrix_savedir=None,
        **kwargs
    ):
        if rotation_angles is None:
            rotation_angles = [0]

        buffer = []
        for theta in rotation_angles:
            for i, ref in enumerate(tqdm(ref_image_dataloader, disable=not verbose)):
                if DEBUG > 1 and i > 10:
                    break
                image = ref["image"].to(device)
                image = TF.rotate(image, theta)
                model_args = {}
                if save_salad_matrix:
                    model_args["return_matrix"] = True
                model_out = model(image, **model_args)
                x = model_out["out"]
                model_theta = model_out.get("theta", None)
                salad_matrix = model_out.get("salad_matrix", None)

                if salad_matrix is not None and save_salad_matrix:
                    assert salad_matrix_savedir is not None, "salad_matrix_savedir must be provided to save salad matrices"
                    self.save_ref_salad_matrix(salad_matrix_savedir, f"salad_matrix_ref{i:05d}.npy", salad_matrix)

                if model_theta is not None:
                    self.theta_buffer.append(model_theta.cpu())
                if self.norm_vec:
                    x = F.normalize(x)
                buffer.append(x)
                if len(buffer) >= buffer_size:
                    self._flush_buffer(buffer)
        self._flush_buffer(buffer)
        if len(self.theta_buffer) > 0:
            self.theta_buffer = torch.cat(self.theta_buffer).numpy()


class Mast3rASMKVectorDatabase:

    def __init__(self, checkpoint_path: str):
        self.asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                        'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                        'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                        'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                        'search': {'topk': None},
                                        'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}
        assert os.path.isdir(checkpoint_path), "`checkpoint_path` must be a directory containing mast3r_retrieval and mast3r_codebook checkpoints"
        retrieval_checkpoint = os.path.join(checkpoint_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
        codebook_checkpoint = os.path.join(checkpoint_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")

        ckpt = torch.load(retrieval_checkpoint, 'cpu', weights_only=False)
        assert os.path.isfile(codebook_checkpoint), codebook_checkpoint
        self.asmk_params['train_codebook']['codebook']['size'] = ckpt["args"].nclusters
        self.asmk = asmk_method.ASMKMethod.initialize_untrained(self.asmk_params)
        self.asmk = self.asmk.train_codebook(None, cache_path=codebook_checkpoint)
        self.asmk_builder = self.asmk.create_ivf_builder(cache_path=None, step_params=None)
        self.asmk_dataset = None
        self.vdim = None

    def add(self, vectors, imids):
        self.asmk_builder.add(vectors, imids)

    def save(self, savedir: str):
        cache_path = os.path.join(savedir, "asmk_index.index")
        io_helpers.save_pickle(cache_path, self.asmk_dataset.inverted_file.state_dict())
        return {**self.asmk_dataset.metadata, "ivf_stats": self.asmk_dataset.inverted_file.stats}
    
    def search(self, qu: torch.Tensor, imids, k: int):
        self.asmk_dataset.params['query_ivf']['search']['topk'] = k
        qu = qu.cpu().numpy().squeeze()
        metadata, query_ids, ranks, ranked_scores = self.asmk_dataset.query_ivf(qu, imids)
        return ranked_scores, ranks
    
    def size(self):
        return self.asmk_dataset.inverted_file.stats["images"]
    
    def _flush_buffer(self, buffer):
        if buffer:
            x_buffer = map(lambda buff: np.concatenate(buff), zip(*buffer))
            self.add(*x_buffer)
            buffer.clear()

    @torch.no_grad()
    def build(
        self,
        ref_image_dataloader,
        model,
        rotation_angles=None,
        device="cpu",
        **kwargs
    ):
        if rotation_angles is None:
            rotation_angles = [0]
        assert ref_image_dataloader.batch_size == 1, (
            "Mast3rASMKVectorDatabase.build() only supports batch_size=1 (because of imids shananigans)"
        )
        for j, theta in enumerate(rotation_angles):
            for i, ref in enumerate(tqdm(ref_image_dataloader)):
                if DEBUG > 1 and i > 10:
                    break
                image = ref["image"].to(device)
                if theta != 0:
                    image = TF.rotate(image, theta)
                _idx = i + j * len(ref_image_dataloader)
                outdict = model(image, idx=_idx)
                x = outdict["out"].cpu().numpy().squeeze()
                imids = outdict["ids"].cpu().numpy().squeeze()
                self.add(x, imids)
        self.asmk_dataset = self.asmk.add_ivf_builder(self.asmk_builder)


def load_database(loaddir: str, config):
    """Load a faiss index from a file."""

    def detect_db_type(config):
        if config.vdb.class_path.endswith("Mast3rASMKVectorDatabase"):
            return "asmk"
        return "faiss"

    db_type = detect_db_type(config)
    db_index_path = os.path.join(loaddir, f"{db_type}_index.index")
    assert os.path.exists(db_index_path), f"Index file not found: {db_index_path}"
    db = class_from_config(config.vdb)

    if db_type == "asmk":
        db.asmk_dataset = db.asmk.build_ivf(cache_path=db_index_path)
    else:
        cpu_index = faiss.read_index(db_index_path)
        if db.faiss_gpu:
            res = faiss.StandardGpuResources()
            db.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            db.faiss_index = cpu_index
    return db
