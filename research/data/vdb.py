import os
import torch
import faiss
from torch.nn import functional as F
import json

class VectorDatabase:

	def __init__(
		self,
		vdim: int,
		distfn: str = "l2",
		norm_vec: bool = True
	):
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
		self.faiss_index.add(vectors)

	def search(self, qu: torch.Tensor, k: int):
		if self.norm_vec:
			qu = F.normalize(qu)
		distances, indices = self.faiss_index.search(qu, k)
		return distances, indices
	
	def size(self):
		return self.faiss_index.ntotal
	
	def save(self, savedir: str):
		index_savepath = os.path.join(savedir, "faiss_index.index")
		faiss.write_index(self.faiss_index, index_savepath)
		db_metadata = {
			"vdim": self.vdim,
			"distfn": self.distfn,
			"norm_vec": self.norm_vec
		}
		metadata_path = os.path.join(savedir, "index_metadata.json")
		with open(metadata_path, "w") as f:
			json.dump(db_metadata, f)


def load_database(loaddir: str):
	"""Load a faiss index from a file."""
	db_metadata_path = os.path.join(loaddir, "index_metadata.json")
	db_index_path = os.path.join(loaddir, "faiss_index.index")

	assert os.path.exists(db_metadata_path), f"Metadata file not found: {db_metadata_path}"
	assert os.path.exists(db_index_path), f"Index file not found: {db_index_path}"

	with open(db_metadata_path, "r") as f:
		db_metadata = json.load(f)	
			
	db = VectorDatabase(
		vdim=db_metadata["vdim"], 
		distfn=db_metadata["distfn"], 
		norm_vec=db_metadata["norm_vec"]
	)
	db.faiss_index = faiss.read_index(db_index_path)
	return db