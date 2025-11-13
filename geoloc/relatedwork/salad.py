import lightning as L
import torch

# from torch.optim import lr_scheduler, optimizer
from torch.optim import lr_scheduler

# from .models import helper
from ..models.aggregators.salad import SALAD
from ..models.backbones import SaladDINOv2Backbone
from ..config_parser import dict_to_namespace
from ..models.vprmodel import VPRModel


class SALADModel(VPRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_pretrained(self, ckpt_path: str):
        self.load_state_dict(torch.load(ckpt_path), strict=True)


# def get_loss(loss_name):
# 	from pytorch_metric_learning import losses
# 	from pytorch_metric_learning.distances import DotProductSimilarity

# 	if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
# 	if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
# 	if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
# 	if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
# 	if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
# 	if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
# 	if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
# 	if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
# 	if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
# 																			swap=False,
# 																			smooth_loss=False,
# 																			triplets_per_anchor="all",)
# 	raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

# def get_miner(miner_name, margin=0.1):
# 	from pytorch_metric_learning import miners
# 	from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

# 	if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
# 	if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
# 	if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
# 	return None


# class SALADModel(L.LightningModule):
# 	"""This is the main model for Visual Place Recognition
# 	we use Pytorch Lightning for modularity purposes.

# 	Args:
# 		pl (_type_): _description_
# 	"""

# 	def __init__(self,
# 		#---- Backbone
# 		model_name="dinov2_vitb14",
# 		num_trainable_blocks=4,
# 		return_token=True,
# 		norm_layer=True,
# 		#---- Aggregator
# 		num_channels=768,
# 		num_clusters=64,
# 		cluster_dim=128,
# 		token_dim=256,
# 		#---- Train hyperparameters
# 		lr=6e-5,
# 		optimizer='adamw',
# 		weight_decay=9.5e-9,
# 		momentum=0.9,
# 		lr_sched='linear',
# 		lr_sched_args = {
# 			'start_factor': 1,
# 			'end_factor': 0.2,
# 			'total_iters': 4000,
# 		},

# 		#----- Loss
# 		loss_name='MultiSimilarityLoss',
# 		miner_name='MultiSimilarityMiner',
# 		miner_margin=0.1,
# 		faiss_gpu=False
# 	):
# 		super().__init__()
# 		# Backbone
# 		self.model_name = model_name
# 		self.num_trainable_blocks = num_trainable_blocks
# 		self.return_token = return_token
# 		self.norm_layer = norm_layer

# 		# Aggregator
# 		self.num_channels = num_channels
# 		self.num_clusters = num_clusters
# 		self.cluster_dim = cluster_dim
# 		self.token_dim = token_dim

# 		# Train hyperparameters
# 		self.lr = lr
# 		self.optimizer = optimizer
# 		self.weight_decay = weight_decay
# 		self.momentum = momentum
# 		self.lr_sched = lr_sched
# 		self.lr_sched_args = lr_sched_args

# 		# Loss
# 		self.loss_name = loss_name
# 		self.miner_name = miner_name
# 		self.miner_margin = miner_margin

# 		self.save_hyperparameters() # write hyperparams into a file

# 		self.loss_fn = get_loss(loss_name)
# 		self.miner = get_miner(miner_name, miner_margin)
# 		self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level

# 		self.faiss_gpu = faiss_gpu

# 		self.backbone = SaladDINOv2Backbone(
# 			model_name=self.model_name,
# 			num_trainable_blocks=self.num_trainable_blocks,
# 			return_token=self.return_token,
# 			norm_layer=self.norm_layer
# 		)
# 		self.aggregator = SALAD(
# 			num_channels=self.num_channels,
# 			num_clusters=self.num_clusters,
# 			cluster_dim=self.cluster_dim,
# 			token_dim=self.token_dim
# 		)

# 		# For validation in Lightning v2.0.0
# 		self.val_outputs = []

# 	# the forward pass of the lightning model
# 	def forward(self, x):
# 		x = self.backbone(x)
# 		x = self.aggregator(x)
# 		return x

# 	# configure the optimizer
# 	def configure_optimizers(self):
# 		if self.optimizer.lower() == 'sgd':
# 			optimizer = torch.optim.SGD(
# 				self.parameters(),
# 				lr=self.lr,
# 				weight_decay=self.weight_decay,
# 				momentum=self.momentum
# 			)
# 		elif self.optimizer.lower() == 'adamw':
# 			optimizer = torch.optim.AdamW(
# 				self.parameters(),
# 				lr=self.lr,
# 				weight_decay=self.weight_decay
# 			)
# 		elif self.optimizer.lower() == 'adam':
# 			optimizer = torch.optim.AdamW(
# 				self.parameters(),
# 				lr=self.lr,
# 				weight_decay=self.weight_decay
# 			)
# 		else:
# 			raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')


# 		if self.lr_sched.lower() == 'multistep':
# 			# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
# 			scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args.milestones, gamma=self.lr_sched_args.gamma)
# 		elif self.lr_sched.lower() == 'cosine':
# 			# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
# 			scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args.T_max)
# 		elif self.lr_sched.lower() == 'linear':
# 			scheduler = lr_scheduler.LinearLR(
# 				optimizer,
# 				# start_factor=self.lr_sched_args['start_factor'],
# 				# end_factor=self.lr_sched_args['end_factor'],
# 				# total_iters=self.lr_sched_args['total_iters']
# 				start_factor=self.lr_sched_args.start_factor,
# 				end_factor=self.lr_sched_args.end_factor,
# 				total_iters=self.lr_sched_args.total_iters
# 			)

# 		return [optimizer], [scheduler]

# 	# configure the optizer step, takes into account the warmup stage
# 	def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
# 		# warm up lr
# 		optimizer.step(closure=optimizer_closure)
# 		self.lr_schedulers().step()

# 	#  The loss function call (this method will be called at each training iteration)
# 	def loss_function(self, descriptors, labels):
# 		# we mine the pairs/triplets if there is an online mining strategy
# 		if self.miner is not None:
# 			miner_outputs = self.miner(descriptors, labels)
# 			loss = self.loss_fn(descriptors, labels, miner_outputs)

# 			# calculate the % of trivial pairs/triplets
# 			# which do not contribute in the loss value
# 			nb_samples = descriptors.shape[0]
# 			nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
# 			batch_acc = 1.0 - (nb_mined/nb_samples)

# 		else: # no online mining
# 			loss = self.loss_fn(descriptors, labels)
# 			batch_acc = 0.0
# 			if type(loss) == tuple:
# 				# somes losses do the online mining inside (they don't need a miner objet),
# 				# so they return the loss and the batch accuracy
# 				# for example, if you are developping a new loss function, you might be better
# 				# doing the online mining strategy inside the forward function of the loss class,
# 				# and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
# 				loss, batch_acc = loss

# 		# keep accuracy of every batch and later reset it at epoch start
# 		self.batch_acc.append(batch_acc)
# 		# log it
# 		self.log('b_acc', sum(self.batch_acc) /
# 				len(self.batch_acc), prog_bar=True, logger=True)
# 		return loss

# 	# This is the training step that's executed at each iteration
# 	def training_step(self, batch, batch_idx):
# 		imgs = batch["images"]
# 		BS, N, ch, h, w = imgs.shape
# 		imgs = imgs.view(BS * N, ch, h, w)
# 		labels = torch.arange(BS).repeat_interleave(N)

# 		# Feed forward the batch to the model
# 		descriptors = self(imgs) # Here we are calling the method forward that we defined above

# 		if torch.isnan(descriptors).any():
# 			raise ValueError('NaNs in descriptors')

# 		loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above

# 		self.log('loss', loss.item(), logger=True, prog_bar=True)
# 		return {'loss': loss}

# 	def on_train_epoch_end(self):
# 		# we empty the batch_acc list for next epoch
# 		self.batch_acc = []

# 	# # For validation, we will also iterate step by step over the validation set
# 	# # this is the way Pytorch Lghtning is made. All about modularity, folks.
# 	# def validation_step(self, batch, batch_idx, dataloader_idx=None):
# 	#     places, _ = batch
# 	#     descriptors = self(places)
# 	#     self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
# 	#     return descriptors.detach().cpu()

# 	# def on_validation_epoch_start(self):
# 	#     # reset the outputs list
# 	#     self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]

# 	# def on_validation_epoch_end(self):
# 	#     """this return descriptors in their order
# 	#     depending on how the validation dataset is implemented
# 	#     for this project (MSLS val, Pittburg val), it is always references then queries
# 	#     [R1, R2, ..., Rn, Q1, Q2, ...]
# 	#     """
# 	#     val_step_outputs = self.val_outputs

# 	#     dm = self.trainer.datamodule
# 	#     # The following line is a hack: if we have only one validation set, then
# 	#     # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
# 	#     if len(dm.val_datasets)==1: # we need to put the outputs in a list
# 	#         val_step_outputs = [val_step_outputs]

# 	#     for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
# 	#         feats = torch.concat(val_step_outputs[i], dim=0)

# 	#         if 'pitts' in val_set_name:
# 	#             # split to ref and queries
# 	#             num_references = val_dataset.dbStruct.numDb
# 	#             positives = val_dataset.getPositives()
# 	#         elif 'msls' in val_set_name:
# 	#             # split to ref and queries
# 	#             num_references = val_dataset.num_references
# 	#             positives = val_dataset.pIdx
# 	#         else:
# 	#             print(f'Please implement validation_epoch_end for {val_set_name}')
# 	#             raise NotImplemented

# 	#         r_list = feats[ : num_references]
# 	#         q_list = feats[num_references : ]
# 	#         pitts_dict = utils.get_validation_recalls(
# 	#             r_list=r_list,
# 	#             q_list=q_list,
# 	#             k_values=[1, 5, 10, 15, 20, 50, 100],
# 	#             gt=positives,
# 	#             print_results=True,
# 	#             dataset_name=val_set_name,
# 	#             faiss_gpu=self.faiss_gpu
# 	#         )
# 	#         del r_list, q_list, feats, num_references, positives

# 	#         self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
# 	#         self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
# 	#         self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
# 	#     print('\n\n')

# 	#     # reset the outputs list
# 	#     self.val_outputs = []
