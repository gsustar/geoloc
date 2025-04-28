import matplotlib.pyplot as plt
from data.dataset_VisLoc import VisLocReferenceImages, VisLocQueryImages

def main():
	root = "/storage/datasets/AerialLoc/UAV_VisLoc"
	ref_image_dataset = VisLocReferenceImages(root=root, flight_idx=1, tile_size=256, stride=256)
	tile = ref_image_dataset[10]["image"]
	plt.imshow(tile.permute(1, 2, 0))  # Transpose to (H, W, C) for plotting
	plt.savefig("tile.png")
	query_image_dataset = VisLocQueryImages(root=root, flight_idx=1, resize=(256, 256))
	query = query_image_dataset[0]["image"]
	plt.imshow(query.permute(1, 2, 0))  # Transpose to (H, W, C) for plotting
	plt.savefig("query.png")

	print("all okay!")

if __name__ == "__main__":
	main()