# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import requests
import os
import time
import yaml
import sys
import json
import mercantile
from tqdm import tqdm
from multiprocessing import Pool
import geopandas as gpd
import random


# Number of workers for parallel processing
NUM_WORKERS = 4
BURST_SIZE = 500
BURST_COUNTER = 0

class DownloaderConfig:
	EXPECTED_SCHEMA = {
		"tiles_dir": str,
		"arcgis": dict,
		"mapbox": dict,
		"ges": dict,
		"uav_dataset_dir": str,
		"sat_zoom_level": int,
		"download_endpoint": str,
		"dataset": str,
		"regions_filepath": str,
	}

	def __init__(self, config_filepath: str = "./downloader_config.yaml"):
		if not os.path.exists(config_filepath):
			raise Exception("Specified config file not found!\nexiting...")

		self.config_filepath = config_filepath
		self.config = self.parse_config(config_filepath)
		self.validate_config()

	def parse_config(self, config_filepath: str):
		with open(config_filepath, "r") as config_file:
			config = yaml.load(config_file, Loader=yaml.FullLoader)
			return config["downloader"]

	def validate_config(self):
		"""
		Validates the parsed configuration against the expected schema.
		"""
		for key, expected_type in self.EXPECTED_SCHEMA.items():
			if key not in self.config:
				raise Exception(f"Missing configuration parameter: {key}")

			if not isinstance(self.config[key], expected_type):
				raise Exception(
					f"Invalid type for {key}. Expected {expected_type}, got {type(self.config[key])}"
				)

		for key in self.config:
			if key not in self.EXPECTED_SCHEMA:
				raise Exception(f"Unexpected configuration parameter: {key}")

		print(f"Configuration is valid! {self.config}")

	def add_endpoints(self, endpoints: dict):
		self.config["download_endpoints"] = endpoints


### SATELLITE TILES ###
def parse_available_maps_endpoint(config: DownloaderConfig) -> None:
	response = requests.get(config.config.get("arcgis").get("arcgis_available_maps_endpoint"))
	root = ET.fromstring(response.text)

	ns = {
		"wmts": "https://www.opengis.net/wmts/1.0",
		"ows": "https://www.opengis.net/ows/1.1",
	}

	filtered_endpoints = {}
	endpoints = []

	for layer in root.findall(".//wmts:Layer", ns):
		title_elem = layer.find("ows:Title", ns)
		tile_name = title_elem.text if title_elem is not None else None

		resource_url_elem = layer.find("wmts:ResourceURL", ns)
		tile_template = (
			resource_url_elem.get("template") if resource_url_elem is not None else None
		)

		if tile_name is None or tile_template is None:
			continue

		tile_name = tile_name.split("(")[1].split(")")[0]
		tile_name = tile_name.split(" ")[1]
		endpoints.append({"name": tile_name, "template": tile_template})

	for year in config.config.get("arcgis").get("endpoints_filter"):
		filtered_endpoints[str(year)] = list(
			endpoint for endpoint in endpoints if str(year) in endpoint["name"]
		)

	config.add_endpoints(filtered_endpoints)


def download_tile(config: DownloaderConfig, zoom: int, x: int, y: int, session: requests.Session = None):
	download_endpoint = config.config.get("download_endpoint")
	if download_endpoint == "arcgis":
		TILE_MATRIX_SET = "default028mm"
		for year in config.config.get("download_endpoints"):
			successfully_downloaded = False
			tile_name = None
			for endpoint in config.config.get("download_endpoints")[year]:
				if successfully_downloaded:
					break
				tiles_dir = config.config.get("tiles_dir").format(download_endpoint=download_endpoint)
				tile_name = tiles_dir + "/" + str(year) + "/zoom" + str(zoom)
				template_url = endpoint["template"]
				tile_path = os.path.join(tile_name, f"{zoom}_{x}_{y}.jpg")

				if os.path.exists(tile_path):
					print(f"Tile {tile_path} already exists, skipping...")
					successfully_downloaded = True
					continue

				url = template_url.format(
					TileMatrixSet=TILE_MATRIX_SET, TileMatrix=zoom, TileRow=y, TileCol=x
				)
				successfully_downloaded = download_with_retry(config, url, zoom, x, y, tile_name, tile_path, session)
	elif download_endpoint == "mapbox":
		this_year = time.strftime("%Y")
		tiles_dir = config.config.get("tiles_dir").format(download_endpoint=download_endpoint)
		tile_name = tiles_dir + "/" + this_year + "/zoom" + str(zoom)
		tile_path = os.path.join(tile_name, f"{zoom}_{x}_{y}.jpg")
		if os.path.exists(tile_path):
			return
		url = config.config.get("mapbox").get("mapbox_endpoint").format(
				x = x, y = y, z = zoom, mapbox_access_token = config.config.get("mapbox").get("mapbox_access_token") 
		)
		download_with_retry(config, url, zoom, x, y, tile_name, tile_path, session)
	elif download_endpoint == "ges":
		this_year = time.strftime("%Y")
		tiles_dir = config.config.get("tiles_dir").format(download_endpoint=download_endpoint)
		tile_name = tiles_dir + "/" + this_year + "/zoom" + str(zoom)
		tile_path = os.path.join(tile_name, f"{zoom}_{x}_{y}.jpg")
		if os.path.exists(tile_path):
			return
		url = config.config.get("ges").get("ges_endpoint").format(
				x = x, y = y, z = zoom
		)
		download_with_retry(config, url, zoom, x, y, tile_name, tile_path, session)
	else:
		print("Invalid download endpoint!")
		exit(42)


def download_with_retry(config: DownloaderConfig, url: str, zoom: int, x: int, y: int, tile_name: str, tile_path: str, session: requests.Session = None):
	max_attempts = 3  # Maximum number of retry attempts
	successfully_downloaded = False
	session = session or requests

	for attempt in range(max_attempts):
		response = None
		try:
			if attempt > 0:
				print(f"Retrying {attempt + 1} of {max_attempts} for tile {tile_name} at {zoom}_{x}_{y}...")
				# Exponential backoff
				jitter = random.uniform(0, 1)
				time.sleep(4 ** attempt + jitter)  # Exponential backoff with jitter
			response = session.get(
				url,
				headers={
					"Referer": "https://livingatlas.arcgis.com/",
					"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
				},
				timeout=10,
			)
			response.raise_for_status()  # raises a Python exception if the response contains an HTTP error status code
		except (
			requests.exceptions.RequestException,
			requests.exceptions.ConnectionError,
		) as e:
			if (
				attempt < max_attempts - 1
			):  # i.e., if it's not the final attempt
				if response is None:
					time.sleep(5)  # wait for 5 seconds before trying again
					continue
				if response.status_code == 404:
					print(f"Endpoint {url} returned 404. Skipping...")
					break
				print(
					f"Attempt {attempt + 1} of {max_attempts} failed for tile {tile_name} at {zoom}_{x}_{y}. Retrying..."
				)
				print("Response: ", response.text)
				time.sleep(5)  # wait for 5 seconds before trying again
				continue
			else:
				print(
					f"Error downloading tile {tile_name} at {zoom}_{x}_{y} from {url}: {e}"
				)
				break
		else:  # executes if the try block didn't throw any exceptions
			os.makedirs(tile_name, exist_ok=True)
			with open(tile_path, "wb") as tile_file:
				tile_file.write(response.content)
			successfully_downloaded = True
			break  # break out of the retry loop as the tile has been successfully downloaded
	if not successfully_downloaded:
		raise Exception(
			f"Failed to download tile {tile_name} at {zoom}_{x}_{y} from any of the available endpoints"
		)
	print(f"Succesfully downloaded tile {tile_name} at {zoom}_{x}_{y} from {url}")
	return successfully_downloaded


def get_tile_from_coord(lat, lng, zoom_level):
	tile = mercantile.tile(lng, lat, zoom_level)
	return tile


### DRONE DATASET ###
def download_uav_pair_sat_images(
	config: DownloaderConfig, latitude: float, longitude: float
):
	"""
	Each tile: 256x256
	Joined tile: 1280x1280
	+--------+--------+--------+--------+--------+
	|        |        |        |        |        |
	|        |        |        |        |        |
	+--------+--------+--------+--------+--------+
	|        |        |        |        |        |
	|        |        |        |        |        |
	+--------+--------+--------+--------+--------+
	|        |        |  x     |        |        |
	|        |        |        |        |        |
	+--------+--------+--------+--------+--------+
	|        |        |        |        |        |
	|        |        |        |        |        |
	+--------+--------+--------+--------+--------+
	|        |        |        |        |        |
	|        |        |        |        |        |
	+--------+--------+--------+--------+--------+
	"""

	# Check if latitude and longitude are floats
	if not isinstance(latitude, float) or not isinstance(longitude, float):
		raise Exception(
			f"Latitude ({latitude}) and longitude ({longitude}) must be floats!, exiting..."
		)

	tile = get_tile_from_coord(
		latitude,
		longitude,
		config.config.get("sat_zoom_level"),
	)
	main_neighbours = mercantile.neighbors(tile)

	all_tiles_to_download = []

	for close_neighbour in main_neighbours:
		for far_neighbour in mercantile.neighbors(close_neighbour):
			if far_neighbour not in all_tiles_to_download:
				all_tiles_to_download.append(far_neighbour)

	for neighbour in all_tiles_to_download:
		download_tile(config, neighbour.z, neighbour.x, neighbour.y)


def validate_dataset(
	config: DownloaderConfig,
	images_path: str,
	metadata_path: str,
):
	def sort_by_last_digits(arr):
		def key_func(s):
			return int(s.split("_")[-1].replace(".jpeg", ""))

		return sorted(arr, key=key_func)

	def get_dataset_name(images_path: str):
		return images_path.split("/")[-2]

	with open(metadata_path, newline="") as jsonfile:
		json_dict = json.load(jsonfile)

		camera_frames = json_dict["cameraFrames"]

		imgs = []
		for filename in os.listdir(images_path):
			if filename.endswith(".jpeg"):
				imgs.append(filename)
			else:
				continue

		pairs = list(zip(sort_by_last_digits(imgs), camera_frames))

		for image_path, corresponding_metadata in tqdm(
			pairs, total=len(pairs), desc=get_dataset_name(images_path)
		):
			image_path = images_path + "/" + image_path

			download_uav_pair_sat_images(
				config,
				corresponding_metadata["coordinate"]["latitude"],
				corresponding_metadata["coordinate"]["longitude"],
			)


def validate_dataset_wrapper(args):
	config, image_path, metadata_path = args
	return validate_dataset(config, image_path, metadata_path)


def iterate_through_datasets_GES(config: DownloaderConfig):
	pairs = []  # this will hold tuples of (image_path, metadata_path)

	for directory in os.listdir(config.config.get("uav_dataset_dir")):
		if "Train" in directory or "Test" in directory:
			image_path = None
			metadata_path = None

			for subdirectory in os.listdir(
				config.config.get("uav_dataset_dir") + "/" + directory
			):
				if "footage" in subdirectory:
					image_path = os.path.join(
						config.config.get("uav_dataset_dir"), directory, subdirectory
					)
				elif ".json" in subdirectory:
					metadata_path = os.path.join(
						config.config.get("uav_dataset_dir"), directory, subdirectory
					)

			if image_path and metadata_path:
				pairs.append((image_path, metadata_path))
			else:
				if not image_path:
					print(f"No images found for {directory}")
				if not metadata_path:
					print(f"No metadata found for {directory}")

	with Pool(os.cpu_count()) as p:
		p.map(validate_dataset_wrapper, [(config, pair[0], pair[1]) for pair in pairs])


def download_uavloc(
	config: DownloaderConfig,
	metadata_path: str,
):

	with open(metadata_path, "r") as f:
		metadata = json.load(f)

	for entry in metadata:
		latitude = entry["content"]["GPS"]["latitude"]
		longitude = entry["content"]["GPS"]["longitude"]
		#try:
		download_uav_pair_sat_images(
				config,
				entry["content"]["GPS"]["latitude"],
				entry["content"]["GPS"]["longitude"],
		)
		#except Exception as e:
		#    print("Error for entry: ", entry, metadata_path)


def download_uavloc_wrapper(args):
	config, metadata_path = args
	download_uavloc(config, metadata_path)


def iterate_through_datasets_UAV_LOC(config: DownloaderConfig):
	metadata_paths = []
	for f in os.listdir(config.config.get("uav_dataset_dir")):
		if "json" in f:
			metadata_path = os.path.join(config.config.get("uav_dataset_dir"), f)
			metadata_paths.append(metadata_path)

	with Pool(os.cpu_count()) as p:
		p.map(
			download_uavloc_wrapper,
			[(config, metadata_path) for metadata_path in metadata_paths],
		)


def download_region_wrapper(args):
	config, zoom, x, y, session = args
	download_tile(config, zoom, x, y, session)


def iterate_through_region(config: DownloaderConfig):
	regions_filepath = config.config.get("regions_filepath")
	zoom = config.config.get("sat_zoom_level")

	gdf = gpd.read_file(regions_filepath)
	# gdf = gdf.to_crs(epsg=4326) # project to lon/lat
	# gdf = gdf.to_crs(epsg=3794)

	# Get bounding box of multipolygon
	lon_min, lat_min, lon_max, lat_max = gdf.total_bounds

	min_tile = get_tile_from_coord(lat_min, lon_min, zoom)
	max_tile = get_tile_from_coord(lat_max, lon_max, zoom)

	x_min = min_tile.x
	y_min = min_tile.y
	x_max = max_tile.x
	y_max = max_tile.y

	session = requests.Session()
	with Pool(processes=NUM_WORKERS) as p:
		p.map(
			download_region_wrapper,
			[(config, zoom, x, y, session) for x in range(min(x_min, x_max), max(x_min, x_max) + 1)
			 for y in range(min(y_min, y_max), max(y_min, y_max) + 1)]
		)

### DRONE DATASET ###

def main(config_filepath: str):
	downloader_config = DownloaderConfig(config_filepath=config_filepath)
	parse_available_maps_endpoint(config=downloader_config)
	if downloader_config.config.get("dataset") == "GES":
		iterate_through_datasets_GES(downloader_config)
	elif downloader_config.config.get("dataset") == "UAV_LOC":
		iterate_through_datasets_UAV_LOC(downloader_config)
	elif downloader_config.config.get("dataset") == "region":
		iterate_through_region(downloader_config)
	else:
		print("Invalid dataset type!")


if __name__ == "__main__":
	if len(sys.argv) == 2:
		main(config_filepath=sys.argv[1])
	else:
		main(config_filepath="./downloader_config.yaml")
