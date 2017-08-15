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


class MemoryLoomLayer():
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


class LoomLayer():
	def __init__(self, ds: loompy.LoomConnection, name: str, dtype: str) -> None:
		self.ds = ds
		self.name = name
		self.dtype = dtype
		self.shape = ds.shape

	def __getitem__(self, slice: Tuple[Union[int, slice], Union[int, slice]]) -> np.ndarray:
		if self.name == "":
			return self.ds._file['/matrix'].__getitem__(slice)
		return self.ds._file['/layers/' + self.name].__getitem__(slice)

	def __setitem__(self, slice: Tuple[Union[int, slice], Union[int, slice]], data: np.ndarray) -> None:
		if self.name == "":
			self.ds._file['/matrix'].__setitem__(slice, data.astype(self.dtype))
		else:
			self.ds._file['/layers/' + self.name].__setitem__(slice, data.astype(self.dtype))

	def sparse(self, genes: np.ndarray, cells: np.ndarray) -> scipy.sparse.coo_matrix:
		n_genes = self.ds.shape[0] if genes is None else genes.shape[0]
		n_cells = self.ds.shape[1] if cells is None else cells.shape[0]
		data: np.ndarray = None
		row: np.ndarray = None
		col: np.ndarray = None
		for (ix, selection, vals) in self.ds.batch_scan(genes=genes, cells=cells, axis=1, layer=self.name):
			nonzeros = np.where(vals > 0)
			if data is None:
				data = vals[nonzeros]
				row = nonzeros[0]
				col = selection[nonzeros[1]]
			else:
				data = np.concatenate([data, vals[nonzeros]])
				row = np.concatenate([row, nonzeros[0]])
				col = np.concatenate([col, selection[nonzeros[1]]])
		return scipy.sparse.coo_matrix((data, (row, col)), shape=(n_genes, n_cells))
		
	def resize(self, size: Tuple[int, int], axis: int = None) -> None:
		"""Resize the dataset, or the specified axis.
		The dataset must be stored in chunked format; it can be resized up to the "maximum shape" (keyword maxshape) specified at creation time.
		The rank of the dataset cannot be changed.
		"Size" should be a shape tuple, or if an axis is specified, an integer.
		BEWARE: This functions differently than the NumPy resize() method!
		The data is not "reshuffled" to fit in the new shape; each axis is grown or shrunk independently.
		The coordinates of existing data are fixed.
		"""
		if self.name == "":
			self.ds._file['/matrix'].resize(size, axis)
		else:
			self.ds._file['/layers/' + self.name].resize(size, axis)

