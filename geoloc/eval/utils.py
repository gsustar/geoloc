import os
from matplotlib.table import table
import prettytable as pt

def bytes_to_gb(bytes: int):
	return bytes / (1024**3)

def format_header(header: str, width: int = 40) -> str:
	header_text = f"== {header.upper()} =="
	return header_text.center(width, "=")


def write_resdict_to_file(file, resdict: dict, header: str = None):
	if header:
		print(format_header(header), file=file)
		print("", file=file)
	for key, value in resdict.items():
		print(f"{key:<25}: {value}", file=file)
	print("\n", file=file)


def write_pretty_table(file, field_names, data, align="l", header: str = None):
	table = pt.PrettyTable()
	table.field_names = field_names
	table.align = align
	for row in data:
		formatted_row = []
		for val in row:
			formatted_row.append(f"{val:.2f}" if isinstance(val, float) else val)
		table.add_row(formatted_row)

	if header:
		print(format_header(header), file=file)
		print("", file=file)
	print(table, file=file)
	print("\n", file=file)