import os
import torch
import faiss
import inspect 
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torchvision.transforms.functional as TF

try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None

try:
    from scipy.spatial import KDTree
except ImportError as e:
    KDTree = None

try:
    from asmk import asmk_method
    from asmk import io_helpers
except ImportError:
    asmk_method = None
    io_helpers = None
    print("Warning: asmk package not found. Mast3rASMKVectorDatabase will not work.")

from ..config_parser import class_from_config
from ..utils import DEBUG, requires_arg


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
        self.imInd_buffer = []
        
        self.spatial_index = None
        self.spatial_coords = None
        self.spatial_crs = None

    def set_spatial_index(self, coords: np.ndarray, crs: str = "EPSG:4326"):
        coords = np.asarray(coords)
        assert coords.ndim == 2 and coords.shape[1] == 2, "`coords` must have shape (N, 2) with [y, x] ordering"
        assert coords.shape[0] == self.size(), (
            f"Number of spatial coordinates ({coords.shape[0]}) must match DB size ({self.size()})"
        )

        self.spatial_coords = coords.astype(np.float64)
        self.spatial_crs = crs

        if crs == "EPSG:4326":
            assert BallTree is not None, "BallTree is required for WGS84 radius search. Install scikit-learn."
            coords_rad = np.radians(self.spatial_coords)
            self.spatial_index = BallTree(coords_rad, metric="haversine")
        else:
            assert KDTree is not None, "KDTree is required for projected radius search. Install scipy."
            self.spatial_index = KDTree(self.spatial_coords)

    def _radius_filter_ids(self, query_y: float, query_x: float, radius_m: float):
        assert self.spatial_index is not None, "Spatial index not initialized. Call `set_spatial_index(...)` first."
        assert self.spatial_coords is not None, "Spatial coordinates not initialized."

        if self.spatial_crs == "WGS84":
            center_rad = np.radians([[query_y, query_x]])
            radius_rad = radius_m / 6_371_000.0
            ids = self.spatial_index.query_radius(center_rad, r=radius_rad)[0]
        else:
            ids = self.spatial_index.query_ball_point([query_y, query_x], r=radius_m)
            ids = np.asarray(ids)

        return ids.astype(np.int64)

    def search_radius(self, qu: torch.Tensor, k: int, query_y: float, query_x: float, radius_m: float):
        assert not self.faiss_gpu, "Radius search with IDSelector is only supported when `faiss_gpu=False`."

        valid_ids = self._radius_filter_ids(query_y=query_y, query_x=query_x, radius_m=radius_m)
        if len(valid_ids) == 0:
            empty_d = np.empty((qu.shape[0], 0), dtype=np.float32)
            empty_i = np.empty((qu.shape[0], 0), dtype=np.int64)
            return empty_d, empty_i

        if self.norm_vec:
            qu = F.normalize(qu)

        selector = faiss.IDSelectorBatch(valid_ids)
        params = faiss.SearchParameters()
        params.sel = selector
        distances, indices = self.faiss_index.search(qu.cpu().numpy(), k, params=params)
        return distances, indices

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
        
        if len(self.imInd_buffer) > 0:
            imInd_savepath = os.path.join(savedir, "segvlad_imInds.npy")
            np.save(imInd_savepath, np.array(self.imInd_buffer))

    def _flush_buffer(self, buffer):
        if buffer:
            x_buffer = torch.vstack(buffer)
            self.add(x_buffer)
            buffer.clear()

    def save_ref_salad_matrix(self, savedir: str, batch_ix: int, salad_matrix: torch.Tensor):
        os.makedirs(savedir, exist_ok=True)
        bs = salad_matrix.shape[0]
        for j in range(bs):
            salad_path = os.path.join(savedir, f"salad_matrix_ref{batch_ix * bs + j:05d}.npy")
            np.save(salad_path, salad_matrix[j].cpu().numpy())

    @torch.no_grad()
    def build(
        self,
        ref_image_dataloader,
        model,
        rotation_angles=None,
        device="cpu",
        buffer_size=6000,
        verbose=True,
        **kwargs
    ):
        if rotation_angles is None:
            rotation_angles = [0]

        buffer = []
        save_salad_matrix = kwargs.get("save_salad_matrix", False)
        salad_matrix_savedir = kwargs.get("salad_matrix_savedir", None)
        requires_idx = requires_arg(model.forward, "idx")

        for theta in rotation_angles:
            for i, ref in enumerate(tqdm(ref_image_dataloader, disable=not verbose)):
                if DEBUG > 1 and i > 10:
                    break
                image = ref["image"].to(device)
                image = TF.rotate(image, theta)

                model_args = {}
                if save_salad_matrix:
                    model_args["return_salad_matrix"] = True
                if requires_idx:
                    model_args["idx"] = i

                model_out = model(image, **model_args)
                x = model_out["out"]

                salad_matrix = model_out.get("salad_matrix", None)
                if salad_matrix is not None and save_salad_matrix:
                    self.save_ref_salad_matrix(salad_matrix_savedir, i, salad_matrix)

                imInds = model_out.get("imInds1_ind", None)
                rgInds = model_out.get("regInds1_ind", None)
                if imInds is not None and rgInds is not None:
                    self.imInd_buffer.append(
                        torch.tensor(list(zip(imInds, rgInds)))
                    )
                model_theta = model_out.get("theta", None)
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
        if len(self.imInd_buffer) > 0:
            self.imInd_buffer = torch.cat(self.imInd_buffer).numpy()


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


class PytorchIndex:
    def __init__(
        self,
        vdim: int,
        distfn: str = "l2",
        norm_vec: bool = True,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        if distfn not in ("cosine", "l2"):
            raise ValueError(f"Invalid distance function: {distfn}")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Invalid dtype: {dtype}")

        self.vdim = vdim
        self.distfn = distfn
        self.norm_vec = norm_vec
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # The index itself: (N, vdim) on self.device. Empty holds no storage.
        self.vectors = torch.empty(0, vdim, device=self.device, dtype=self.dtype)
        self.vdbdir = None

        self.spatial_index = None
        self.spatial_coords = None
        self.spatial_crs = None

    def size(self):
        return self.vectors.shape[0]

    def add(self, vectors: torch.Tensor):
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim != 2 or vectors.shape[1] != self.vdim:
            raise ValueError(
                f"Expected vectors of shape (N, {self.vdim}), got {tuple(vectors.shape)}"
            )
        if self.norm_vec:
            vectors = F.normalize(vectors)
        vectors = vectors.to(self.device, self.dtype)
        self.vectors = torch.cat((self.vectors, vectors), dim=0)

    def set_spatial_index(self, coords: np.ndarray, crs: str = "EPSG:4326"):
        coords = np.asarray(coords)
        assert coords.ndim == 2 and coords.shape[1] == 2, "`coords` must have shape (N, 2) with [y, x] ordering"
        assert coords.shape[0] == self.size(), (
            f"Number of spatial coordinates ({coords.shape[0]}) must match DB size ({self.size()})"
        )

        self.spatial_coords = coords.astype(np.float64)
        self.spatial_crs = crs

        if crs == "EPSG:4326":
            assert BallTree is not None, "BallTree is required for WGS84 radius search. Install scikit-learn."
            coords_rad = np.radians(self.spatial_coords)
            self.spatial_index = BallTree(coords_rad, metric="haversine")
        else:
            assert KDTree is not None, "KDTree is required for projected radius search. Install scipy."
            self.spatial_index = KDTree(self.spatial_coords)

    def _radius_filter_ids(self, query_y: float, query_x: float, radius_m: float):
        assert self.spatial_index is not None, "Spatial index not initialized. Call `set_spatial_index(...)` first."
        assert self.spatial_coords is not None, "Spatial coordinates not initialized."

        if self.spatial_crs == "WGS84":
            center_rad = np.radians([[query_y, query_x]])
            radius_rad = radius_m / 6_371_000.0
            ids = self.spatial_index.query_radius(center_rad, r=radius_rad)[0]
        else:
            ids = self.spatial_index.query_ball_point([query_y, query_x], r=radius_m)
            ids = np.asarray(ids)

        return ids.astype(np.int64)

    def search_radius(self, qu: torch.Tensor, k: int, query_y: float, query_x: float, radius_m: float):
        if qu.ndim == 1:
            qu = qu.unsqueeze(0)

        valid_ids = self._radius_filter_ids(query_y=query_y, query_x=query_x, radius_m=radius_m)
        if len(valid_ids) == 0:
            empty_d = np.empty((qu.shape[0], 0), dtype=np.float32)
            empty_i = np.empty((qu.shape[0], 0), dtype=np.int64)
            return empty_d, empty_i

        ids = torch.from_numpy(valid_ids).to(self.device)
        distances, indices = self._search(qu, k, self.vectors[ids])
        return distances, valid_ids[indices]

    def search(self, qu: torch.Tensor, k: int):
        return self._search(qu, k, self.vectors)

    def _search(self, qu: torch.Tensor, k: int, vectors: torch.Tensor):
        if qu.ndim == 1:
            qu = qu.unsqueeze(0)
        if qu.ndim != 2 or qu.shape[1] != self.vdim:
            raise ValueError(
                f"Expected queries of shape (M, {self.vdim}), got {tuple(qu.shape)}"
            )
        if self.norm_vec:
            qu = F.normalize(qu)
        qu = qu.to(self.device, self.dtype)

        k = min(k, vectors.shape[0])
        if self.distfn == "cosine":
            ip = vectors @ qu.T
            distances, indices = ip.topk(k, dim=0)
            distances, indices = distances.T, indices.T
        else:
            d = torch.cdist(qu, vectors)
            distances, indices = d.topk(k, dim=1, largest=False)

        return distances.float().contiguous().cpu().numpy(), indices.contiguous().cpu().numpy()

    @torch.no_grad()
    def build(self, ref_image_dataloader, model, device="cpu", verbose=True, **kwargs):
        for i, ref in enumerate(tqdm(ref_image_dataloader, disable=not verbose)):
            if DEBUG > 1 and i > 10:
                break
            image = ref["image"].to(device)
            emb = model(image)["out"]
            if self.norm_vec:
                emb = F.normalize(emb)
            self.add(emb)

    def save(self, savedir: str):
        self.vdbdir = savedir
        os.makedirs(savedir, exist_ok=True)
        torch.save(self.vectors.cpu(), os.path.join(savedir, "pytorch_index.index"))


def load_database(loaddir: str, config):
    """Load a faiss index from a file."""

    def detect_db_type(config):
        if config.vdb.class_path.endswith("Mast3rASMKVectorDatabase"):
            return "asmk"
        if config.vdb.class_path.endswith("PytorchIndex"):
            return "pytorch"
        return "faiss"

    db_type = detect_db_type(config)
    db_index_path = os.path.join(loaddir, f"{db_type}_index.index")
    assert os.path.exists(db_index_path), f"Index file not found: {db_index_path}"
    db = class_from_config(config.vdb)

    if db_type == "asmk":
        db.asmk_dataset = db.asmk.build_ivf(cache_path=db_index_path)
    elif db_type == "pytorch":
        vectors = torch.load(db_index_path, weights_only=True)
        assert vectors.shape[1] == db.vdim, (
            f"Saved index has vdim {vectors.shape[1]}, config says {db.vdim}"
        )
        db.vectors = vectors.to(db.device, db.dtype)
    else:
        cpu_index = faiss.read_index(db_index_path)
        if db.faiss_gpu:
            res = faiss.StandardGpuResources()
            db.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            db.faiss_index = cpu_index
    return db

def convert_flat_index_to_pytorch(index_path: str, save_path: str):
    index = faiss.read_index(index_path)
    thindex = torch.from_numpy(index.reconstruct_n(0, index.ntotal))
    torch.save(thindex, save_path)
