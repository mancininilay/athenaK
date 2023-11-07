#! /usr/bin/env python3
import struct
import warnings
import numpy as np

import matplotlib.pyplot as plt

class BinaryData:
  """
    This class represents a binary data dump from AthenaK.

    The AthenaK binary file is structured as follows:
    Header:
      Athena binary output version=1.1
        size of preheader=5
        time=<time here>
        cycle=<cycle here>
        size of location=<Real size here>
        size of variable=<float size here>
        number of variables=<#>
        variables:  <var1>  <var2>  <var3>  ...
        #------------------------- PAR_DUMP -------------------------
        <block>
        par = value
        #------------------------- PAR_DUMP -------------------------
        <par_end>
        header offset=<size of header in bytes>
    Per MeshBlock:
      <ois><oie><ojs><oje><oks><oke> // MeshBlock indexing (int32_t)
      <lx1><lx2><lx3>                // Logical location (int32_t)
      <refinement level>             // Physical refinement level (int32_t)
      <x1min><x1max><x2min><x2max><x3min><x3max> // Coordinate location (Real)
      <var1(0,0,0)><var1(0,0,1)>...
      ...
      <varn(0,0,0)><varn(0,0,1)>...  // Output variables (float)
  """
  def __init__(self, filename):
    # Try to read the file metadata
    with open(filename, 'rb') as f:
      #print(f'Opening file {filename}...')
      self.filename = filename
      # Get the file size
      f.seek(0, 2)
      self.file_size = f.tell()
      f.seek(0, 0)

      #print(f'Extracting metadata...')
      # Read header metadata
      line = f.readline().decode('ascii')
      if line != 'Athena binary output version=1.1\n':
        raise RuntimeError('Unrecognized data file format.')
      next(f)
      # Read in time
      line = f.readline().decode('ascii')
      self.time = float(line[7:])
      next(f)
      # Get size of Real
      line = f.readline().decode('ascii')
      if line[:19] != '  size of location=':
        raise RuntimeError('Could not read location size.')
      self.location_size = int(line[19:])
      # Get size of variable (usually float)
      line = f.readline().decode('ascii')
      if line[:19] != '  size of variable=':
        raise RuntimeError('Could not read variable size.')
      self.variable_size = int(line[19:])
      # Get number of variables
      next(f)
      line = f.readline().decode('ascii')
      if line[:12] != '  variables:':
        raise RuntimeError('Could not read variable names.')
      self.variable_names = line[12:].split()
      # Get header offset in bytes
      line = f.readline().decode('ascii')
      if line[:16] != '  header offset=':
        raise RuntimeError('Could not read header offset.')
      self.header_offset = int(line[16:])

      # Process metadata
      if self.location_size not in (4, 8):
        raise RuntimeError('Only 4- and 8-byte integer types supported for '
                           'location data.')
      location_format = 'f' if self.location_size == 4 else 'd'
      if self.variable_size not in (4, 8):
        raise RuntimeError('Only 4- and 8-byte integer types supported for cell '
                           'data.')
      variable_format = 'f' if self.variable_size == 4 else 'd'

      #print(f'Reading input file...')

      # Read in input file
      self.input_file = {}
      self.start_of_data = f.tell() + self.header_offset
      while f.tell() < self.start_of_data:
        line = f.readline().decode('ascii')
        if line[0] == '#':
          continue
        if line[0] == '<':
          section_name = line[1:-2]
          self.input_file[section_name] = {}
          continue
        key, val = line.split('=', 1)
        self.input_file[section_name][key.strip()] = val.split('#', 1)[0].strip()

      # Lists to hold results
      mesh = self.input_file['mesh']
      nghost = int(mesh['nghost'])
      self.extent = [float(mesh['x1min']), float(mesh['x1max']),
                     float(mesh['x2min']), float(mesh['x2max']),
                     float(mesh['x3min']), float(mesh['x3max'])]
      self.blocks = []
      cells_per_block = -1
      varsize = -1
      nvars = len(self.variable_names)

      #print(f'Reading grid structure...')

      # Read in grid structure data
      first_time = True
      while f.tell() < self.file_size:
        block_indices = np.array(struct.unpack('@6i', f.read(24))) - nghost
        block_i, block_j, block_k, block_level = struct.unpack('@4i', f.read(16))

        # Process grid structure
        if first_time:
          block_nx = block_indices[1] - block_indices[0] + 1
          block_ny = block_indices[3] - block_indices[2] + 1
          block_nz = block_indices[5] - block_indices[4] + 1
          cells_per_block = block_nz * block_ny * block_nx
          self.block_cell_format = '=' + str(cells_per_block) + variable_format
          varsize = cells_per_block * self.variable_size
          first_time = False

        # Read in coordinate data
        block_lims = struct.unpack('=6' + location_format, f.read(6 * self.location_size))

        self.blocks.append(MeshBlock([block_nx, block_ny, block_nz],
                           varsize, block_lims,
                           self.variable_names, f.tell()))
        f.seek(nvars*varsize,1)
  
    self.derived_vars = {}

  def is_2d(self):
    return self.blocks[0].is_2d()

  def plot_slice(self, var, ax=plt, cmap='viridis', norm=None, vmin=None, vmax=None,
                 interpolation='none', origin='lower', slice_loc=None,rescale=1.):
    pcm = None
    if len(var) >= 8:
      if var[:8] == 'derived:':
        if self.is_2d():
          with open(self.filename, 'rb') as f:
            fn = self.derived_vars[var][0]
            fnvars = self.derived_vars[var][1:]

            for block in self.blocks:
              fnargs = []
              for v in fnvars:
                fnargs.append(block.get_var(f, v, self.block_cell_format)*rescale)
              data = fn(*tuple(fnargs))
              extent = block.get_extent()
              pcm = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                         interpolation=interpolation, origin=origin,
                         extent=extent)
        return pcm

    if not self.is_2d() and slice_loc==None:
      if slice_loc==None:
        raise RuntimeError('Slice must be specified for 3D data.')
    elif self.is_2d():
      if slice_loc!=None:
        raise RuntimeError('Slice cannot be specified for 2D data.')
      with open(self.filename, 'rb') as f:
        for block in self.blocks:
          data = block.get_var(f, var, self.block_cell_format)*rescale
          extent = block.get_extent()
          pcm = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                     interpolation=interpolation, origin=origin,
                     extent=extent)
    return pcm

  def register_derived_variable(self, name, f, *args):
    if len(name) < 8:
      raise RuntimeError("Derived variable name must begin with 'derived:'!")
    elif not name[:8] == 'derived:':
      raise RuntimeError("Derived variable name must begin with 'derived:'!")
    self.derived_vars[name] = [f, *args]

class MeshBlock:
  '''
  It's often the case that the files we're trying to process are too large to store
  entirely in memory. If we instead store the MeshBlocks with the location of each
  variable in the file, we can stream it off the disk instead.
  '''
  def __init__(self, size, datasize, coords, variables, datastart):
    self.size = size
    self.datasize = datasize
    self.coords = coords
    self.dx = [(coords[1]-coords[0])/size[0],
               (coords[3]-coords[2])/size[1],
               (coords[5]-coords[4])/size[2]]
    self.datastart = datastart

    self.quantities = {}
    count = datastart
    for v in variables:
      self.quantities[v] = count
      count += datasize

  def get_var(self, f, name, block_cell_format, slice_mask = None):
    f.seek(self.quantities[name], 0)
    nz = self.size[2]
    ny = self.size[1]
    nx = self.size[0]
    data = (np.array(struct.unpack(block_cell_format, f.read(self.datasize)))
            .reshape(nz, ny, nx))
    if self.is_2d():
      if nx == 1:
        return data[:,:,0]
      elif ny == 1:
        return data[:,0,:]
      else:
        return data[0,:,:]
    else:
      if slice_mask == None:
        return data
      else:
        return data[slice_mask]

  def get_extent(self):
    if self.is_2d():
      if self.size[0] == 1:
        return (self.coords[2], self.coords[3], self.coords[4], self.coords[5])
      if self.size[1] == 1:
        return (self.coords[0], self.coords[1], self.coords[4], self.coords[5])
      if self.size[2] == 1:
        return (self.coords[0], self.coords[1], self.coords[2], self.coords[3])
    else:
      return (self.coords[0], self.coords[1], self.coords[2], self.coords[3],
              self.coords[4], self.coords[5])

  def get_coord_blocks(self):
    if self.is_2d():
      xf = None
      yf = None
      if self.size[0] == 1:
        xf = np.linspace(self.coords[2], self.coords[3], self.size[1] + 1)
        yf = np.linspace(self.coords[4], self.coords[5], self.size[2] + 1)
      elif self.size[1] == 1:
        xf = np.linspace(self.coords[0], self.coords[1], self.size[0] + 1)
        yf = np.linspace(self.coords[4], self.coords[5], self.size[2] + 1)
      elif self.size[2] == 1:
        xf = np.linspace(self.coords[0], self.coords[1], self.size[0] + 1)
        yf = np.linspace(self.coords[2], self.coords[3], self.size[1] + 1)
      xc = 0.5*(xf[:-1] + xf[1:])
      yc = 0.5*(yf[:-1] + yf[1:])

      return np.meshgrid(yc, xc, indexing='ij')
    else:
      xf = np.linspace(self.coords[0], self.coords[1], self.size[0] + 1)
      yf = np.linspace(self.coords[2], self.coords[3], self.size[1] + 1)
      zf = np.linspace(self.coords[4], self.coords[5], self.size[2] + 1)
      xc = 0.5*(xf[:-1] + xf[1:])
      yc = 0.5*(yf[:-1] + yf[1:])
      zc = 0.5*(zf[:-1] + zf[1:])

      return np.meshgrid(z, y, x, indexing='ij')

  def is_in_slice(self, dimension, x):
    if dimension == 'x':
      return (self.coords[0] <= x and self.coords[1] >= x)
    elif dimension == 'y':
      return (self.coords[2] <= x and self.coords[3] >= x)
    elif dimension == 'z':
      return (self.coords[4] <= x and self.coords[5] >= x)
    else:
      raise RuntimeError('Invalid slice dimension')

  def make_slice_mask(self, coord_block, dimension, s):
    dx = 0.
    if dimension == 'x':
      dx = self.dx[0]
    elif dimension == 'y':
      dx = self.dx[1]
    elif dimension == 'z':
      dx = self.dx[2]
    else:
      raise RuntimeError('Invalid slice dimension')
    
    return (coord_block >= (s - 0.5*dx)) & (coord_block <= (s + 0.5*dx))

  def is_2d(self):
    return (self.size[0] == 1) or (self.size[1] == 1) or (self.size[2] == 1)
