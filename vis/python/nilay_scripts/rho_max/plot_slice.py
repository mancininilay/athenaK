#! /usr/bin/env python3

"""
Script for plotting a 2D slice from a 2D or 3D AthenaK data dump.

Usage:
[python3] plot_slice.py <input_file> <quantity_to_plot> <output_file> [options]

Example:
~/athenak/vis/python/plot_slice.py basename.prim.00100.bin dens image.png

<input_file> can be any standard AthenaK .bin data dump. <output_file> can have
any extension recognized by Matplotlib (e.g., .png). If <output_file> is simply
"show", the script will open a live Matplotlib viewing window.

Available quantities include anything found in the input file (e.g., dens, velx,
eint, bcc1, r00_ff). If an invalid quantity is requested (e.g.,
"plot_slice.py <input_file> ? show"), the error message will list all available
quantities in the file.

Additional derived quantities can be computed as well. These are identified by
prefixing with "derived:". An invalid request (e.g.,
"plot_slice.py <input_file> derived:? show") will list available options.
Currently, these include the following:
  - Quantities related to gas pressure:
    - pgas: gas pressure
    - kappa: gas pressure divided by density^gamma
    - betath: gas pressure divided by total pressure (define k tilde and gamma!!!!!!!!!!!)
    - epsilon: conserved energy divided by density
    - T: temperature
  - Non-relativistic quantities related to magnetic pressure:
    - pmag_nr: (magnetic pressure) = B^2 / 2
    - beta_inv_nr: 1 / (plasma beta) = (magnetic pressure) / (gas pressure)
    - sigma_nr: (plasma sigma) = B^2 / rho
  - Relativistic quantities related to magnetic pressure:
    - pmag_rel: (magnetic pressure) = (B^2 - E^2) / 2
    - beta_inv_rel: 1 / (plasma beta) = (magnetic pressure) / (gas pressure)
    - sigma_rel: (plasma sigma) = (B^2 - E^2) / rho
  - Relativistic quantities related to radiation:
    - prad: (radiation pressure) = (fluid-frame radiation energy density) / 3
    - prad_pgas: (radiation pressure) / (gas pressure)
    - pmag_prad: (magnetic pressure) / (radiation pressure)
  - Relativistic quantities related to velocity:
    - uut: normal-frame Lorentz factor u^{t'} = tilde{u}^t
    - ut, ux, uy, uz: contravariant coordinate-frame 4-velocity components u^mu
    - vx, vy, vz: coordinate-frame 3-velocity components u^i / u^t
  - Non-relativistic conserved quantities
    - cons_hydro_nr_t: pure hydrodynamical energy density
    - cons_hydro_nr_x, cons_hydro_nr_y, cons_hydro_nr_z: momentum density
    - cons_em_nr_t: pure electromagnetic energy density
    - cons_mhd_nr_t: MHD energy density
    - cons_mhd_nr_x, cons_mhd_nr_y, cons_mhd_nr_z: MHD momentum density
  - Relativistic conserved quantities
    - cons_hydro_rel_t, : (T_hydro)^t_t
    - cons_hydro_rel_x, cons_hydro_rel_y, cons_hydro_rel_z: (T_hydro)^t_i
    - cons_em_rel_t, : (T_EM)^t_t
    - cons_em_rel_x, cons_em_rel_y, cons_em_rel_z: (T_EM)^t_i
    - cons_mhd_rel_t, : (T_MHD)^t_t
    - cons_mhd_rel_x, cons_mhd_rel_y, cons_mhd_rel_z: (T_MHD)^t_i

Only temperature T is in physical units (K); all others are in code units.

Optional inputs include:
  -d: direction orthogonal to slice of 3D data
  -l: location of slice of 3D data if not 0
  --x1_min, --x1_max, --x2_min, --x2_max: horizontal and vertical limits of plot
  -c: colormap recognized by Matplotlib
  -n: colormap normalization (e.g., "-n log") if not linear
  --vmin, --vmax: limits of colorbar if not the full range of data
  --notex: flag to disable Latex typesetting of labels
  --horizon: flag for outlining outer event horizon of GR simulation
  --horizon_mask: flag for covering black hole of GR simulation
  --ergosphere: flag for outlining boundary of ergosphere in GR simulation
  --horizon_color, --horizon_mask_color, --ergosphere_color: color choices

Run "plot_slice.py -h" to see a full description of inputs.
"""

# Python standard modules
import argparse
import struct
import warnings

# Numerical modules
import numpy as np

# Load plotting modules
import matplotlib


# Main function
def main(**kwargs):

    # Load additional numerical modules
    if kwargs['ergosphere']:
        from scipy.optimize import brentq

    # Load additional plotting modules
    if kwargs['output_file'] != 'show':
        matplotlib.use('agg')
    if not kwargs['notex']:
        matplotlib.rc('text', usetex=True)
    import matplotlib.colors as colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # Plotting parameters
    horizon_line_style = '-'
    horizon_line_width = 1.0
    ergosphere_num_points = 129
    ergosphere_line_style = '-'
    ergosphere_line_width = 1.0
    x1_labelpad = 2.0
    x2_labelpad = 2.0
    dpi = 300

    # Adjust user inputs
    if kwargs['dimension'] == '1':
        kwargs['dimension'] = 'x'
    if kwargs['dimension'] == '2':
        kwargs['dimension'] = 'y'
    if kwargs['dimension'] == '3':
        kwargs['dimension'] = 'z'

    # Set physical units
    c_cgs = 2.99792458e10
    kb_cgs = 1.380649e-16
    mp_cgs = 1.67262192369e-24
    gg_msun_cgs = 1.32712440018e26

    # Set derived dependencies
    derived_dependencies = {}
    derived_dependencies['pgas'] = ('eint',)
    derived_dependencies['epsilon'] = ('dens', 'ener')
    derived_dependencies['kappa'] = ('dens', 'eint')
    derived_dependencies['betath'] = ('dens', 'eint')
    derived_dependencies['T'] = ('dens', 'eint')
    derived_dependencies['pthermal'] = ('dens', 'eint')
    derived_dependencies['pmag_nr'] = ('bcc1', 'bcc2', 'bcc3')
    derived_dependencies['pmag_rel'] = ('velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_nr'] = ('eint', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_rel'] = ('eint', 'velx', 'vely', 'velz',
                                            'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['sigma_nr'] = ('dens', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['sigma_rel'] = ('dens', 'velx', 'vely', 'velz',
                                         'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['prad'] = ('r00_ff',)
    derived_dependencies['prad_pgas'] = ('eint', 'r00_ff')
    derived_dependencies['pmag_prad'] = ('velx', 'vely', 'velz',
                                         'bcc1', 'bcc2', 'bcc3', 'r00_ff')
    derived_dependencies['uut'] = ('velx', 'vely', 'velz')
    derived_dependencies['ut'] = ('velx', 'vely', 'velz')
    derived_dependencies['ux'] = ('velx', 'vely', 'velz')
    derived_dependencies['uy'] = ('velx', 'vely', 'velz')
    derived_dependencies['uz'] = ('velx', 'vely', 'velz')
    derived_dependencies['vx'] = ('velx', 'vely', 'velz')
    derived_dependencies['vy'] = ('velx', 'vely', 'velz')
    derived_dependencies['vz'] = ('velx', 'vely', 'velz')
    derived_dependencies['cons_hydro_nr_t'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['cons_hydro_nr_x'] = ('dens', 'velx')
    derived_dependencies['cons_hydro_nr_y'] = ('dens', 'vely')
    derived_dependencies['cons_hydro_nr_z'] = ('dens', 'velz')
    derived_dependencies['cons_em_nr_t'] = ('bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_nr_t'] = ('dens', 'eint', 'velx', 'vely', 'velz',
                                             'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_nr_x'] = ('dens', 'velx')
    derived_dependencies['cons_mhd_nr_y'] = ('dens', 'vely')
    derived_dependencies['cons_mhd_nr_z'] = ('dens', 'velz')
    derived_dependencies['cons_hydro_rel_t'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['cons_hydro_rel_x'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['cons_hydro_rel_y'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['cons_hydro_rel_z'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['cons_em_rel_t'] = ('velx', 'vely', 'velz',
                                             'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_em_rel_x'] = ('velx', 'vely', 'velz',
                                             'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_em_rel_y'] = ('velx', 'vely', 'velz',
                                             'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_em_rel_z'] = ('velx', 'vely', 'velz',
                                             'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_rel_t'] = ('dens', 'eint', 'velx', 'vely', 'velz',
                                              'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_rel_x'] = ('dens', 'eint', 'velx', 'vely', 'velz',
                                              'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_rel_y'] = ('dens', 'eint', 'velx', 'vely', 'velz',
                                              'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['cons_mhd_rel_z'] = ('dens', 'eint', 'velx', 'vely', 'velz',
                                              'bcc1', 'bcc2', 'bcc3')

    # Read data
    with open(kwargs['data_file'], 'rb') as f:

        # Get file size
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        # Read header metadata
        line = f.readline().decode('ascii')
        if line != 'Athena binary output version=1.1\n':
            raise RuntimeError('Unrecognized data file format.')
        next(f)
        next(f)
        next(f)
        line = f.readline().decode('ascii')
        if line[:19] != '  size of location=':
            raise RuntimeError('Could not read location size.')
        location_size = int(line[19:])
        line = f.readline().decode('ascii')
        if line[:19] != '  size of variable=':
            raise RuntimeError('Could not read variable size.')
        variable_size = int(line[19:])
        next(f)
        line = f.readline().decode('ascii')
        if line[:12] != '  variables:':
            raise RuntimeError('Could not read variable names.')
        variable_names_base = line[12:].split()
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
            raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        # Process header metadata
        if location_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for '
                               'location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for cell '
                               'data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables_base = len(variable_names_base)
        if kwargs['variable'][:8] == 'derived:':
            variable_name = kwargs['variable'][8:]
            if variable_name not in derived_dependencies:
                raise RuntimeError('Derived variable "{0}" not valid; options are " \
                        "{{{1}}}.'.format(variable_name,
                                          ', '.join(derived_dependencies.keys())))
            variable_names = []
            variable_inds = []
            for dependency in derived_dependencies[variable_name]:
                if dependency not in variable_names_base:
                    raise RuntimeError('Requirement "{0}" for "{1}" not found.'
                                       .format(dependency, variable_name))
                variable_names.append(dependency)
                variable_ind = 0
                while variable_names_base[variable_ind] != dependency:
                    variable_ind += 1
                variable_inds.append(variable_ind)
        elif kwargs['variable'] == 'level':
            variable_name = kwargs['variable']
            variable_names = [variable_name]
            variable_inds = [-1]
        else:
            variable_name = kwargs['variable']
            if variable_name not in variable_names_base:
                raise RuntimeError('Variable "{0}" not found; options are {{{1}}}.'
                                   .format(variable_name,
                                           ', '.join(variable_names_base)))
            variable_names = [variable_name]
            variable_ind = 0
            while variable_names_base[variable_ind] != variable_name:
                variable_ind += 1
            variable_inds = [variable_ind]
        variable_names_sorted = \
            [name for _, name in sorted(zip(variable_inds, variable_names))]
        variable_inds_sorted = \
            [ind for ind, _ in sorted(zip(variable_inds, variable_names))]

        # Read input file metadata
        input_data = {}
        start_of_data = f.tell() + header_offset
        while f.tell() < start_of_data:
            line = f.readline().decode('ascii')
            if line[0] == '#':
                continue
            if line[0] == '<':
                section_name = line[1:-2]
                input_data[section_name] = {}
                continue
            key, val = line.split('=', 1)
            input_data[section_name][key.strip()] = val.split('#', 1)[0].strip()

        # Extract number of ghost cells from input file metadata
        try:
            num_ghost = int(input_data['mesh']['nghost'])
        except:  # noqa: E722
            raise RuntimeError('Unable to find number of ghost cells in input file.')

        # Extract adiabatic index from input file metadata
        if kwargs['variable'] in \
                ['derived:' + name for name in ('pgas', 'kappa','betath', 'T', 'prad_pgas','pthermal')] \
                + ['derived:cons_hydro_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            try:
                gamma_adi = float(input_data['hydro']['gamma'])
            except:  # noqa: E722
                try:
                    gamma_adi = float(input_data['mhd']['gamma'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find adiabatic index in input file.')
        if kwargs['variable'] in \
                ['derived:' + name for name in ('beta_inv_nr', 'beta_inv_rel')] \
                + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            try:
                gamma_adi = float(input_data['mhd']['gamma'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find adiabatic index in input file.')

        # Check input file metadata for relativity
        if kwargs['variable'] in \
                ['derived:' + name for name in ('pmag_nr', 'beta_inv_nr', 'sigma_nr')] \
                + ['derived:cons_hydro_nr_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_em_nr_t'] \
                + ['derived:cons_mhd_nr_' + name for name in ('t', 'x', 'y', 'z')]:
            assert input_data['coord']['general_rel'] == 'false', \
                    '"{0}" is only defined for non-GR data.'.format(variable_name)
        if kwargs['variable'] in \
                ['derived:' + name for name in
                 ('pmag_rel', 'beta_inv_rel', 'sigma_rel', 'pmag_prad')] \
                + ['derived:' + name for name in
                   ('uut', 'ut', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz')] \
                + ['derived:cons_hydro_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_em_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            assert input_data['coord']['general_rel'] == 'true', \
                    '"{0}" is only defined for GR data.'.format(variable_name)
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            assert input_data['coord']['general_rel'] == 'true', '"horizon", ' \
                    '"horizon_mask", and "ergosphere" options only pertain to GR data.'

        # Extract black hole spin from input file metadata
        if kwargs['variable'] in \
                ['derived:' + name for name in
                 ('pmag_rel', 'beta_inv_rel', 'sigma_rel', 'pmag_prad')] \
                + ['derived:' + name for name in
                   ('uut', 'ut', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz')] \
                + ['derived:cons_hydro_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_em_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')

        # Prepare lists to hold results
        max_level_calculated = -1
        block_loc_for_level = []
        block_ind_for_level = []
        num_blocks_used = 0
        extents = []
        quantities = {}
        for name in variable_names_sorted:
            quantities[name] = []

        # Go through blocks
        first_time = True
        while f.tell() < file_size:

            # Read grid structure data
            block_indices = np.array(struct.unpack('@6i', f.read(24))) - num_ghost
            block_i, block_j, block_k, block_level = struct.unpack('@4i', f.read(16))

            # Process grid structure data
            if first_time:
                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                cells_per_block = block_nz * block_ny * block_nx
                block_cell_format = '=' + str(cells_per_block) + variable_format
                variable_data_size = cells_per_block * variable_size
                if kwargs['dimension'] is None:
                    if block_nx > 1 and block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_ny > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_nz > 1:
                        kwargs['dimension'] = 'y'
                    elif block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'x'
                    else:
                        raise RuntimeError('Input file only contains 1D data.')
                if kwargs['dimension'] == 'x':
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_ny
                    block_nx2 = block_nz
                    slice_block_n = block_nx
                    slice_location_min = float(input_data['mesh']['x1min'])
                    slice_location_max = float(input_data['mesh']['x1max'])
                    slice_root_blocks = (int(input_data['mesh']['nx1'])
                                         // int(input_data['meshblock']['nx1']))
                if kwargs['dimension'] == 'y':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_nz
                    slice_block_n = block_ny
                    slice_location_min = float(input_data['mesh']['x2min'])
                    slice_location_max = float(input_data['mesh']['x2max'])
                    slice_root_blocks = (int(input_data['mesh']['nx2'])
                                         // int(input_data['meshblock']['nx2']))
                if kwargs['dimension'] == 'z':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_ny
                    slice_block_n = block_nz
                    slice_location_min = float(input_data['mesh']['x3min'])
                    slice_location_max = float(input_data['mesh']['x3max'])
                    slice_root_blocks = (int(input_data['mesh']['nx3'])
                                         // int(input_data['meshblock']['nx3']))
                slice_normalized_coord = (kwargs['location'] - slice_location_min) \
                    / (slice_location_max - slice_location_min)
                first_time = False

            # Determine if block is needed
            if block_level > max_level_calculated:
                for level in range(max_level_calculated + 1, block_level + 1):
                    if kwargs['location'] <= slice_location_min:
                        block_loc_for_level.append(0)
                        block_ind_for_level.append(0)
                    elif kwargs['location'] >= slice_location_max:
                        block_loc_for_level.append(slice_root_blocks - 1)
                        block_ind_for_level.append(slice_block_n - 1)
                    else:
                        slice_mesh_n = slice_block_n * slice_root_blocks * 2 ** level
                        mesh_ind = int(slice_normalized_coord * slice_mesh_n)
                        block_loc_for_level.append(mesh_ind // slice_block_n)
                        block_ind_for_level.append(mesh_ind - slice_block_n
                                                   * block_loc_for_level[-1])
                max_level_calculated = block_level
            if kwargs['dimension'] == 'x' and block_i != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'y' and block_j != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'z' and block_k != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            num_blocks_used += 1

            # Read coordinate data
            block_lims = struct.unpack('=6' + location_format, f.read(6 * location_size))
            if kwargs['dimension'] == 'x':
                extents.append((block_lims[2], block_lims[3], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'y':
                extents.append((block_lims[0], block_lims[1], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'z':
                extents.append((block_lims[0], block_lims[1], block_lims[2],
                                block_lims[3]))

            # Read cell data
            cell_data_start = f.tell()
            for ind, name in zip(variable_inds_sorted, variable_names_sorted):
                if ind == -1:
                    if kwargs['dimension'] == 'x':
                        quantities[name].append(np.full((block_nz, block_ny),
                                                        block_level))
                    if kwargs['dimension'] == 'y':
                        quantities[name].append(np.full((block_nz, block_nx),
                                                        block_level))
                    if kwargs['dimension'] == 'z':
                        quantities[name].append(np.full((block_ny, block_nx),
                                                        block_level))
                else:
                    f.seek(cell_data_start + ind * variable_data_size, 0)
                    cell_data = (np.array(struct.unpack(block_cell_format,
                                                        f.read(variable_data_size)))
                                 .reshape(block_nz, block_ny, block_nx))
                    block_ind = block_ind_for_level[block_level]
                    if kwargs['dimension'] == 'x':
                        quantities[name].append(cell_data[:, :, block_ind])
                    if kwargs['dimension'] == 'y':
                        quantities[name].append(cell_data[:, block_ind, :])
                    if kwargs['dimension'] == 'z':
                        quantities[name].append(cell_data[block_ind, :, :])
            f.seek((num_variables_base - ind - 1) * variable_data_size, 1)

    # Prepare to calculate derived quantity
    for name in variable_names_sorted:
        quantities[name] = np.array(quantities[name])

    # Calculate derived quantity related to gas pressure
    if kwargs['variable'] in \
            ['derived:' + name for name in ('pgas', 'kappa','betath', 'T', 'pthermal', 'prad_pgas')]:
        pgas = (gamma_adi - 1.0) * quantities['eint']
        ktilde  = 86841
        gamma = 3.005
        conv = (12/11)*(5.635e38/1.6e-6)*(1/8.56e31)
        if kwargs['variable'] == 'derived:pgas':
            quantity = pgas
        elif kwargs['variable'] == 'derived:kappa':
            quantity = pgas / (quantities['dens']**(gamma_adi))
        elif kwargs['variable'] == 'derived:betath':
            quantity = (quantities['eint']- (ktilde*(quantities['dens']**gamma)))/ (quantities['eint'])
        elif kwargs['variable'] == 'derived:pthermal':
            quantity = (quantities['eint']- (ktilde*(quantities['dens']**gamma)))
        elif kwargs['variable'] == 'derived:T':
            quantity = ((quantities['eint']- (ktilde*(quantities['dens']**gamma)))*conv)**0.25 #T in Mev
        else:
            prad = quantities['r00_ff'] / 3.0
            quantity = prad / pgas

    # Calculate derived quantity related to radiation pressure
    elif kwargs['variable'] == 'derived:prad':
        quantity = quantities['r00_ff'] / 3.0
    elif kwargs['variable'] == 'derived:epsilon':
        quantity = quantities['ener'] / quantities['dens']

    # Calculate derived quantity related to non-relativistic magnetic pressure
    elif kwargs['variable'] in \
            ['derived:' + name for name in ('pmag_nr', 'beta_inv_nr', 'sigma_nr')]:
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        pmag = 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)
        if kwargs['variable'] == 'derived:pmag_nr':
            quantity = pmag
        elif kwargs['variable'] == 'derived:beta_inv_nr':
            pgas = (gamma_adi - 1.0) * quantities['eint']
            quantity = pmag / pgas
        else:
            quantity = 2.0 * pmag / quantities['dens']

    # Calculate derived quantity related to relativistic magnetic pressure
    elif kwargs['variable'] in ['derived:' + name for name in
                                ('pmag_rel', 'beta_inv_rel', 'sigma_rel', 'pmag_prad')]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                      kwargs['dimension'], kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz,
                                                   ut, ux, uy, uz, u_x, u_y, u_z)
        b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        pmag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
        if kwargs['variable'] == 'derived:pmag_rel':
            quantity = pmag
        elif kwargs['variable'] == 'derived:beta_inv_rel':
            pgas = (gamma_adi - 1.0) * quantities['eint']
            quantity = pmag / pgas
        elif kwargs['variable'] == 'derived:sigma_rel':
            quantity = 2.0 * pmag / quantities['dens']
        else:
            prad = quantities['r00_ff'] / 3.0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        message=('divide by zero '
                                                 'encountered in true_divide'),
                                        category=RuntimeWarning)
                quantity = pmag / prad

    # Calculate derived quantity related to relativistic velocity
    elif kwargs['variable'] in ['derived:' + name for name in
                                ('uut', 'ut', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz')]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                      kwargs['dimension'], kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        if kwargs['variable'] == 'derived:uut':
            quantity = uut
        elif kwargs['variable'] == 'derived:ut':
            quantity = ut
        elif kwargs['variable'] == 'derived:ux':
            quantity = ux
        elif kwargs['variable'] == 'derived:uy':
            quantity = uy
        elif kwargs['variable'] == 'derived:uz':
            quantity = uz
        elif kwargs['variable'] == 'derived:vx':
            quantity = ux / ut
        elif kwargs['variable'] == 'derived:vy':
            quantity = uy / ut
        else:
            quantity = uz / ut

    # Calculate derived quantity related to non-relativistic conserved variables
    elif kwargs['variable'] in \
            ['derived:cons_hydro_nr_' + name for name in ('t', 'x', 'y', 'z')] \
            + ['derived:cons_em_nr_t'] \
            + ['derived:cons_mhd_nr_' + name for name in ('t', 'x', 'y', 'z')]:
        if kwargs['variable'] in \
                ['derived:cons_' + name + '_nr_t' for name in ('hydro', 'mhd')]:
            rho = quantities['dens']
            ugas = quantities['eint']
            vx = quantities['velx']
            vy = quantities['vely']
            vz = quantities['velz']
            quantity = 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2) + ugas
        elif kwargs['variable'] in \
                ['derived:cons_' + name + '_nr_x' for name in ('hydro', 'mhd')]:
            rho = quantities['dens']
            vx = quantities['velx']
            quantity = rho * vx
        elif kwargs['variable'] in \
                ['derived:cons_' + name + '_nr_y' for name in ('hydro', 'mhd')]:
            rho = quantities['dens']
            vy = quantities['vely']
            quantity = rho * vy
        elif kwargs['variable'] in \
                ['derived:cons_' + name + '_nr_z' for name in ('hydro', 'mhd')]:
            rho = quantities['dens']
            vz = quantities['velz']
            quantity = rho * vz
        else:
            quantity = 0.0
        if kwargs['variable'] in \
                ['derived:cons_' + name + '_nr_t' for name in ('em', 'mhd')]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            quantity += 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)

    # Calculate derived quantity related to relativistic conserved variables
    elif kwargs['variable'] in \
            ['derived:cons_hydro_rel_' + name for name in ('t', 'x', 'y', 'z')] \
            + ['derived:cons_em_rel_' + name for name in ('t', 'x', 'y', 'z')] \
            + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                      kwargs['dimension'], kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        if kwargs['variable'] in \
                ['derived:cons_hydro_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            rho = quantities['dens']
            ugas = quantities['eint']
            pgas = (gamma_adi - 1.0) * ugas
            wgas = rho + ugas + pgas
            if kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_t' for name in ('hydro', 'mhd')]:
                quantity = wgas * ut * u_t + pgas
            elif kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_x' for name in ('hydro', 'mhd')]:
                quantity = wgas * ut * u_x
            elif kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_y' for name in ('hydro', 'mhd')]:
                quantity = wgas * ut * u_y
            else:
                quantity = wgas * ut * u_z
        else:
            quantity = 0.0
        if kwargs['variable'] in \
                ['derived:cons_em_rel_' + name for name in ('t', 'x', 'y', 'z')] \
                + ['derived:cons_mhd_rel_' + name for name in ('t', 'x', 'y', 'z')]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz,
                                                       u_x, u_y, u_z)
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            umag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
            pmag = umag
            if kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_t' for name in ('em', 'mhd')]:
                quantity += (umag + pmag) * ut * u_t + pmag - bt * b_t
            elif kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_x' for name in ('em', 'mhd')]:
                quantity += (umag + pmag) * ut * u_x - bt * b_x
            elif kwargs['variable'] in \
                    ['derived:cons_' + name + '_rel_y' for name in ('em', 'mhd')]:
                quantity += (umag + pmag) * ut * u_y - bt * b_y
            else:
                quantity += (umag + pmag) * ut * u_z - bt * b_z

    # Extract quantity without derivation
    else:
        quantity = quantities[variable_name]

   
    # Prepare figure
    
    print(np.max(quantity))


# Parse inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='name of input file, possibly including path')
    parser.add_argument('variable', help='name of variable to be plotted, any valid'
                                         'derived quantity prefaced by "derived:"')
    args = parser.parse_args()
    main(**vars(args))
