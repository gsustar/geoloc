from PIL import Image
import os
import sys
import json
import mercantile

# TODO: combine tiles with overlapping areas / stride

def find_tiles_bounds(tile_dir):
	# Find the bounds of the tiles
	min_x, min_y = sys.maxsize, sys.maxsize
	max_x, max_y = 0, 0
	for f in os.listdir(tile_dir):
		f = os.path.splitext(f)[0]
		zoom, x, y = os.path.basename(f).split("_")
		x = int(x)
		y = int(y)
		min_x = min(min_x, x)
		min_y = min(min_y, y)
		max_x = max(max_x, x)
		max_y = max(max_y, y)
	return min_x, min_y, max_x, max_y, zoom


def combine_tiles(tile_dir, output_dir, num_tiles_x=4, num_tiles_y=4, tile_size=256, save_stiched_image=True):

	min_x, min_y, max_x, max_y, z = find_tiles_bounds(tile_dir)

	# Calculate the output image size
	width = num_tiles_x * tile_size
	height = num_tiles_y * tile_size
	bounds_min = mercantile.bounds(min_x, min_y, int(z))
	bounds_max = mercantile.bounds(max_x, max_y, int(z))
	area_bounds = {
		"lu": {
			"lon": bounds_min.west,
			"lat": bounds_max.north,
		},
		"rb": {
			"lon": bounds_max.east,
			"lat": bounds_min.south,
		},
		"center": {
			"lon": (bounds_min.west + bounds_max.east) / 2,
			"lat": (bounds_min.north + bounds_max.south) / 2,
		},
	}

	saved_ix = 0
	metadata = {
		"tile_dir": tile_dir,
		"output_dir": output_dir,
		"num_tiles_x": num_tiles_x,
		"num_tiles_y": num_tiles_y,
		"tile_size": tile_size,
		"zoom": int(z),
		"image_size": {
			"width": width,
			"height": height,
		},
		"bbox_area_size": {
			"width": max_x - min_x + 1,
			"height": max_y - min_y + 1,
		},
		"bbox_area_bounds": area_bounds,
		"images": {},
	}

	for x in range(min_x, max_x + 1, num_tiles_x):
		for y in range(min_y, max_y + 1, num_tiles_y):
			combined_image = Image.new("RGB", (width, height))
			tile_bounds = []
			tiles_used = []
			tile_not_found = False

			for i in range(num_tiles_x):
				for j in range(num_tiles_y):
					# Construct the tile filename
					tile_filename = f"{z}_{x+i}_{y+j}.jpg"
					tile_path = os.path.join(tile_dir, tile_filename)
					if os.path.exists(tile_path):
						# Open the tile image
						tile = Image.open(tile_path)
						# Paste the tile into the combined image
						combined_image.paste(tile, (i * tile_size, j * tile_size))
						# Save tile metadata
						tile_bounds.append(mercantile.bounds(x+i, y+j, int(z)))
						tiles_used.append(os.path.join(tile_dir, tile_filename))
					else:
						tile_not_found = True

				if tile_not_found:
					print(f"Tile not found: {tile_path}")
					break

			if tile_not_found:
				continue

			# Compute geographic bounds
			west = min(b.west for b in tile_bounds)
			south = min(b.south for b in tile_bounds)
			east = max(b.east for b in tile_bounds)
			north = max(b.north for b in tile_bounds)
			center_lon = (west + east) / 2
			center_lat = (south + north) / 2

			# Save stitched image
			output_path = os.path.join(output_dir, f"{saved_ix:05d}.jpg")
			saved_ix += 1
			if save_stiched_image:
				combined_image.save(output_path)
			print(f"Saved stitched image to {output_path}")

			# Save metadata
			metadata["images"][output_path] = {
				"lu": {"lon": west, "lat": north},
				"rb": {"lon": east, "lat": south},
				"center": {"lon": center_lon, "lat": center_lat},
				"tiles_used": tiles_used,
			}	
	metadata["num_images"] = saved_ix

	out_meta_path = os.path.join(output_dir, f"metadata.json")
	with open(out_meta_path, "w") as f:
		json.dump(metadata, f, indent=2)
	print(f"Saved metadata to {out_meta_path}")


def construct_image_from_metadata(metadata, image_path, tile_dir):
	""" This function is used to construct images on-the-fly when the `combine_tiles` only saves the metadata.
	"""
	num_tiles_x = metadata["num_tiles_x"]
	num_tiles_y = metadata["num_tiles_y"]
	tile_size = metadata["tile_size"]
	tiles_used = metadata["images"][image_path]["tiles_used"]
	image_size = metadata["image_size"]["width"], metadata["image_size"]["height"]
	combined_image = Image.new("RGB", image_size)
	for i in range(num_tiles_x):
		for j in range(num_tiles_y):
			tile_path = tiles_used[i * num_tiles_y + j]
			tile = Image.open(tile_path)
			combined_image.paste(tile, (i * tile_size, j * tile_size))
	return combined_image


if __name__ == "__main__":
	num_tiles_x = 4
	num_tiles_y = 4
	tile_size = 256
	output_size = (num_tiles_x * tile_size, num_tiles_y * tile_size)

	#* arcgis
	# tiles_dir = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/arcgis/tiles/2025/zoom19"
	# output_dir = f"/storage/datasets/AerialLoc/Drone2Sat/FRISat/arcgis/combined/{output_size[0]}x{output_size[1]}/2025/zoom19/"

	tiles_dir = "/storage/datasets/AerialLoc/Drone2Sat/LjubljanaSat/arcgis/tiles/2025/zoom19"
	output_dir = f"/storage/datasets/AerialLoc/Drone2Sat/LjubljanaSat/arcgis/combined/{output_size[0]}x{output_size[1]}/2025/zoom19/"

	#* ges
	# tiles_dir = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/ges/tiles/zoom19"
	# output_dir = f"/storage/datasets/AerialLoc/Drone2Sat/FRISat/ges/combined/{output_size[0]}x{output_size[1]}/zoom19/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"Created output directory: {output_dir}")

	combine_tiles(tiles_dir, output_dir, num_tiles_x, num_tiles_y, tile_size, save_stiched_image=False)