import numpy as np
from typing import *
import h5py
import os.path
from scipy.io import mmread
import scipy.sparse
from shutil import copyfile
import logging
import time
import loompy


class LoomView:
	"""
	An in-memory loom dataset
	"""
	def __init__(self, layers: Dict[str, MemoryLoomLayer], row_attrs: Dict[str, np.ndarray], col_attrs: Dict[str, np.ndarray]) -> None:
		self.layers = layers
		self.shape = next(iter(self.layers.values())).shape
		self.row_attrs = row_attrs
		self.col_attrs = col_attrs

	def __getitem__(self, slice: Tuple[Union[int, slice], Union[int, slice]]) -> np.ndarray:
		"""
		Get a slice of the main matrix.

		Args:
			slice:		A slice object (see http://docs.h5py.org/en/latest/high/dataset.html)

		Returns:
			A numpy matrix
		"""
		return self.layers[""][slice]
	
	def _repr_html_(self) -> str:
		"""
		Return an HTML representation of the loom view, showing the upper-left 10x10 corner.
		"""
		rm = min(10, self.shape[0])
		cm = min(10, self.shape[1])
		html = "<p>"
		html += "(" + str(self.shape[0]) + " genes, " + str(self.shape[1]) + " cells, " + str(len(self.layers)) + " layers)<br/>"
		html += "<table>"
		# Emit column attributes
		for ca in self.col_attrs.keys():
			html += "<tr>"
			for ra in self.row_attrs.keys():
				html += "<td>&nbsp;</td>"  # Space for row attrs
			html += "<td><strong>" + ca + "</strong></td>"  # Col attr name
			for v in self.col_attrs[ca][:cm]:
				html += "<td>" + str(v) + "</td>"
			if self.shape[1] > cm:
				html += "<td>...</td>"
			html += "</tr>"

		# Emit row attribute names
		html += "<tr>"
		for ra in self.row_attrs.keys():
			html += "<td><strong>" + ra + "</strong></td>"  # Row attr name
		html += "<td>&nbsp;</td>"  # Space for col attrs
		for v in range(cm):
			html += "<td>&nbsp;</td>"
		if self.shape[1] > cm:
			html += "<td>...</td>"
		html += "</tr>"

		# Emit row attr values and matrix values
		for row in range(rm):
			html += "<tr>"
			for ra in self.row_attrs.keys():
				html += "<td>" + str(self.row_attrs[ra][row]) + "</td>"
			html += "<td>&nbsp;</td>"  # Space for col attrs

			for v in self[row, :cm]:
				html += "<td>" + str(v) + "</td>"
			if self.shape[1] > cm:
				html += "<td>...</td>"
			html += "</tr>"
		# Emit ellipses
		if self.shape[0] > rm:
			html += "<tr>"
			for v in range(rm + 1 + len(self.row_attrs.keys())):
				html += "<td>...</td>"
			if self.shape[1] > cm:
				html += "<td>...</td>"
			html += "</tr>"
		html += "</table>"
		return html


class MemoryLoomLayer(LoomLayer):
	def __init__(self, name: str, matrix: np.ndarray) -> None:
		self.name = name
		self.shape = matrix.shape
		self.values = matrix

	def __getitem__(self, slice: Tuple[Union[int, slice], Union[int, slice]]) -> np.ndarray:
		return self.values[slice]

	def __setitem__(self, slice: Tuple[Union[int, slice], Union[int, slice]], data: np.ndarray) -> None:
		self.values[slice] = data

	def sparse(self, genes: np.ndarray, cells: np.ndarray) -> scipy.sparse.coo_matrix:
		return scipy.sparse.coo_matrix(self.values[genes, :][:, cells])
