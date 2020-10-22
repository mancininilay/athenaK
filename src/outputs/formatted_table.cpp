//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formatted_table.cpp
//  \brief writes output data as a formatted (ASCI) table.  Since outputing data in this
//  format is very slow and creates large files, it cannot be used for anything other than
//  1D slices.  Code will issue error if this format is selected for 2D or 3D outputs.
//  TODO: Uses MPI-IO to write all MeshBlocks to a single file

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "utils/grid_locations.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls OutputType base class constructor

FormattedTableOutput::FormattedTableOutput(OutputParameters op, Mesh *pm)
  : OutputType(op, pm)
{
  // check that 1D slice specified, otherwise issue warning and quit
  if (pm->nx2gt1) {
    if (!(out_params.slice1) && !(out_params.slice2)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Formatted table outputs can only contain 1D slices"
                << std::endl << "Please add additional slice planes" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  if (pm->nx3gt1) {
    if ((!(out_params.slice2) && !(out_params.slice3)) ||
        (!(out_params.slice1) && !(out_params.slice3))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Formatted table outputs can only contain 1D slices"
                << std::endl << "Please add additional slice planes" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FormattedTableOutput:::WriteOutputFile(Mesh *pm)
//  \brief writes output_data_ to file in tabular format using C style std::fprintf

void FormattedTableOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
  // create filename: "file_basename" + "." + "file_id" + "." + XXXXX + ".tab"
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".tab");

  // master rank creates file and writes header (even though it may not have any actual
  // data to write below)
  if (global_variable::my_rank == 0) {
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Output file '" << fname << "' could not be opened" << std::endl;
      exit(EXIT_FAILURE);
    }

    // print file header
    std::fprintf(pfile, "# Athena++ data at time=%e", pm->time);
    std::fprintf(pfile, "  cycle=%d \n", pm->ncycle);

    // write one of x1, x2, x3 column headers
    std::fprintf(pfile, "# gid  ");
    if (!(out_params.slice1)) std::fprintf(pfile, " i       x1v     ");
    if (!(out_params.slice2)) std::fprintf(pfile, " j       x2v     ");
    if (!(out_params.slice3)) std::fprintf(pfile, " k       x3v     ");

    // write data col headers from out_data_label_ vector
    for (auto it : out_data_label_) {
      std::fprintf(pfile, "    %s     ", it.c_str());
    }
    std::fprintf(pfile, "\n"); // terminate line
    std::fclose(pfile);   // don't forget to close the output file
  }
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

  // now all ranks open file and append data
  FILE *pfile;
  if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Output file '" << fname << "' could not be opened" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int r=0; r<global_variable::nranks; ++r) {
    // MPI ranks append data one-at-a-time in order, thanks to MPI_Barrier at end of loop
    if (r == global_variable::my_rank) {
      // loop over output MeshBlocks, output all data
      int nout_mbs = static_cast<int>(out_data_.size());
      for (int m=0; m<nout_mbs; ++m) {
        MeshBlock* pmb = pm->FindMeshBlock(out_data_gid_[m]);
        int &is = pmb->mb_cells.is;
        int &js = pmb->mb_cells.js;
        int &ks = pmb->mb_cells.ks;
        Real &x1min = pmb->mb_size.x1min, &x1max = pmb->mb_size.x1max;
        Real &x2min = pmb->mb_size.x2min, &x2max = pmb->mb_size.x2max;
        Real &x3min = pmb->mb_size.x3min, &x3max = pmb->mb_size.x3max;
        int &nx1 = pmb->mb_cells.nx1;
        int &nx2 = pmb->mb_cells.nx2;
        int &nx3 = pmb->mb_cells.nx3;
        for (int k=oks; k<=oke; ++k) {
          for (int j=ojs; j<=oje; ++j) {
            for (int i=ois; i<=oie; ++i) {
              std::fprintf(pfile, "%05d", pmb->mb_gid);
              // write x1, x2, x3 indices and coordinates
              if (oie != ois) {
                std::fprintf(pfile, " %04d", i);
                Real x1cc = CellCenterX(i-is,nx1,x1min,x1max);
                std::fprintf(pfile, out_params.data_format.c_str(), x1cc);
              }
              if (oje != ojs) {
                std::fprintf(pfile, " %04d", j);  // note extra space for formatting
                Real x2cc = CellCenterX(j-js,nx2,x2min,x2max);
                std::fprintf(pfile, out_params.data_format.c_str(), x2cc);
              }
              if (oke != oks) {
                std::fprintf(pfile, " %04d", k);  // note extra space for formatting
                Real x3cc = CellCenterX(k-ks,nx3,x3min,x3max);
                std::fprintf(pfile, out_params.data_format.c_str(), x3cc);
              }

              // write each output variable on same line
              for (int n=0; n<nvar; ++n) {
                std::fprintf(pfile, out_params.data_format.c_str(),
                             out_data_[m](n,k-oks,j-ojs,i-ois));
              }
              std::fprintf(pfile,"\n"); // terminate line
            }
          }
        }
      }  // end loop over MeshBlocks
    }
    std::fflush(pfile);
#if MPI_PARALLEL_ENABLED
    int ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  std::fclose(pfile);   // don't forget to close the output file

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  // store filenumber and time into ParameterInput for restarts
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
