//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file elliptica_sns.cpp
//  \brief Problem generator for single neutron star. Only works when ADM is enabled.

#include <stdio.h>
#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iostream>   // endl
#include <iomanip>
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "elliptica_id_reader_lib.h"


// Prototypes for functions used internally in this pgen.


KOKKOS_INLINE_FUNCTION
static Real A1(Real x1, Real x2, Real x3, Real r0, Real b_norm);
KOKKOS_INLINE_FUNCTION
static Real A2(Real x1, Real x2, Real x3, Real r0, Real b_norm);


// Prototypes for user-defined BCs and history
void TOVHistory(HistoryData *pdata, Mesh *pm);
void VacuumBC(Mesh *pm);
void neutrinolightbulb(Mesh* pm, const Real bdt);
void tovFluxes(HistoryData *pdata, Mesh *pm);


Real C; //heating constant
Real B; //cooling constant
Real rho_cut;  //cutoff for neutrino lightbulb
//Real rho_cut2; //cutoff for hydro freezing
//Real rho_cut3; //cutoff for mhd freezing
Real T; //neutrino temperature
Real Kappatilde; //kappa constant of the true Eos (needed to compute some hydro quantities)
Real Gamma; //nuclear eos adiabatic index
Real b_norm; //magnetic field normalization
Real r0; //magnetic field scale radius
Real m_; //adm mass of the star



//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for TOV star in DynGR

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Single neutron star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }


  user_srcs_func = neutrinolightbulb;
  user_hist_func = tovFluxes;

  auto &grids = spherical_grids;
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 10.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 20.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 30.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 50.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 70.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 100.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 125.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 150.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 200.0));
  //grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 230.0));
  

  // Select either Hydro or MHD
  int nvars;
  std::string block;
  if (pmbp->phydro != nullptr) {
 //   u0_ = pmbp->phydro->u0;
 //   w0_ = pmbp->phydro->w0;
    block = std::string("hydro");
  } else if (pmbp->pmhd != nullptr) {
  //  u0_ = pmbp->pmhd->u0;
  //  w0_ = pmbp->pmhd->w0;
    nvars = pmbp->pmhd->nmhd;
    block = std::string("mhd");
  }

  Real Yfloor = pin->GetOrAddReal(block, "s1_atmosphere", 0.46);
  C = pin->GetOrAddReal("problem", "C", 0.0);
  B = pin->GetOrAddReal("problem", "B", 0.0);
  rho_cut = pin->GetOrAddReal("problem", "rho_cut", 1.0);
  //rho_cut2 = pin->GetOrAddReal("problem", "rho_cut2", 1.0);
  //rho_cut3 = pin->GetOrAddReal("problem", "rho_cut3", 1.0);
  T = pin->GetOrAddReal("problem", "T", 0.0); 
  Kappatilde = pin->GetOrAddReal("problem", "Kappatilde",86841);
  Gamma = pin->GetOrAddReal("problem", "Gamma", 3.005);
  m_ = pin->GetOrAddReal("problem", "m", 1.4);


  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;

  std::string fname = pin->GetString("problem", "initial_data_file");
  
  // Initialize the data reader
  Elliptica_ID_Reader_T *idr = elliptica_id_reader_init(fname.c_str(),"generic");

  // Fields to interpolate
  idr->ifields = "alpha,betax,betay,betaz,"
                 "adm_gxx,adm_gxy,adm_gxz,adm_gyy,adm_gyz,adm_gzz,"
                 "adm_Kxx,adm_Kxy,adm_Kxz,adm_Kyy,adm_Kyz,adm_Kzz,"
                 "grhd_rho,grhd_p,grhd_vx,grhd_vy,grhd_vz";
  

  int width = nmb*ncells1*ncells2*ncells3;

  Real *x_coords = new Real[width];
  Real *y_coords = new Real[width];
  Real *z_coords = new Real[width];

  printf("Allocated coordinates of size %d\n",width);

  // Populate coordinates for Elliptica
  // TODO(JMF): Replace with a Kokkos loop on Kokkos::DefaultHostExecutionSpace() to
  // improve performance.
  int idx = 0;
  for (int m = 0; m < nmb; m++) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    int nx2 = indcs.nx2;

    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    int nx3 = indcs.nx3;

    for (int k = 0; k < ncells3; k++) {
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      for (int j = 0; j < ncells2; j++) {
        Real y = CellCenterX(j - js, nx2, x2min, x2max);
        for (int i = 0; i < ncells1; i++) {
          Real x = CellCenterX(i - is, nx1, x1min, x1max);
          
          x_coords[idx] = x;
          y_coords[idx] = y;
          z_coords[idx] = z;

          // Increment flat index
          idx++;
        }
      }
    }
  }

  idr->set_param("ADM_B1I_form","zero",idr);   //????

  // Interpolate the data
  idr->npoints  = width;
  idr->x_coords = x_coords;
  idr->y_coords = y_coords;
  idr->z_coords = z_coords;
  printf("Coordinates assigned.\n");
  elliptica_id_reader_interpolate(idr);

  // Free the coordinates, since we'll no longer need them.
  delete[] x_coords;
  delete[] y_coords;
  delete[] z_coords;

  printf("Coordinates freed.\n");

  // Capture variables for kernel; note that when Z4c is enabled, the gauge variables
  // are part of the Z4c class.
  auto &u_adm = pmbp->padm->u_adm;
  auto &adm   = pmbp->padm->adm;
  auto &w0    = pmbp->pmhd->w0;
  //auto &u_z4c = pmbp->pz4c->u0;

  // Because Elliptica only operates on the CPU, we can't construct the data on the GPU.
  // Instead, we create a mirror guaranteed to be on the CPU, populate the data there,
  // then move it back to the GPU.
  // TODO(JMF): This needs to be tested on CPUs to ensure that it functions properly;
  // In theory, create_mirror_view shouldn't copy the data unless it's in a different
  // memory space.
  
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = create_mirror_view(w0);
  //HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_BETAX, adm::ADM::I_ADM_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  printf("Host mirrors created.\n");

  // Save Elliptica field indices for shorthand and a small optimization.
  const int i_alpha = idr->indx("alpha");
  const int i_betax = idr->indx("betax");
  const int i_betay = idr->indx("betay");
  const int i_betaz = idr->indx("betaz");

  const int i_gxx   = idr->indx("adm_gxx");
  const int i_gxy   = idr->indx("adm_gxy");
  const int i_gxz   = idr->indx("adm_gxz");
  const int i_gyy   = idr->indx("adm_gyy");
  const int i_gyz   = idr->indx("adm_gyz");
  const int i_gzz   = idr->indx("adm_gzz");

  const int i_Kxx   = idr->indx("adm_Kxx");
  const int i_Kxy   = idr->indx("adm_Kxy");
  const int i_Kxz   = idr->indx("adm_Kxz");
  const int i_Kyy   = idr->indx("adm_Kyy");
  const int i_Kyz   = idr->indx("adm_Kyz");
  const int i_Kzz   = idr->indx("adm_Kzz");

  const int i_rho   = idr->indx("grhd_rho");
  const int i_p     = idr->indx("grhd_p");
  const int i_vx    = idr->indx("grhd_vx");
  const int i_vy    = idr->indx("grhd_vy");
  const int i_vz    = idr->indx("grhd_vz");

  printf("Label indices saved.\n");

  // TODO(JMF): Replace with a Kokkos loop on Kokkos::DefaultHostExecutionSpace() to
  // improve performance.
  idx = 0;
  for (int m = 0; m < nmb; m++) {
    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
          // Extract metric quantities
          host_adm.alpha(m, k, j, i) = idr->field[i_alpha][idx];
          host_adm.beta_u(m, 0, k, j, i) = idr->field[i_betax][idx];
          host_adm.beta_u(m, 1, k, j, i) = idr->field[i_betay][idx];
          host_adm.beta_u(m, 2, k, j, i) = idr->field[i_betaz][idx];
          
          Real g3d[NSPMETRIC];
          host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = idr->field[i_gxx][idx];
          host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = idr->field[i_gxy][idx];
          host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = idr->field[i_gxz][idx];
          host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = idr->field[i_gyy][idx];
          host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = idr->field[i_gyz][idx];
          host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = idr->field[i_gzz][idx];

          host_adm.vK_dd(m, 0, 0, k, j, i) = idr->field[i_Kxx][idx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = idr->field[i_Kxy][idx];
          host_adm.vK_dd(m, 0, 2, k, j, i) = idr->field[i_Kxz][idx];
          host_adm.vK_dd(m, 1, 1, k, j, i) = idr->field[i_Kyy][idx];
          host_adm.vK_dd(m, 1, 2, k, j, i) = idr->field[i_Kyz][idx];
          host_adm.vK_dd(m, 2, 2, k, j, i) = idr->field[i_Kzz][idx];

          // Extract hydro quantities
          host_w0(m, IDN, k, j, i) = idr->field[i_rho][idx];
          host_w0(m, IPR, k, j, i) = idr->field[i_p][idx];
          Real vu[3] = {idr->field[i_vx][idx],
                        idr->field[i_vy][idx],
                        idr->field[i_vz][idx]};

          // Before we store the velocity, we need to make sure it's physical and 
          // calculate the Lorentz factor. If the velocity is superluminal, we make a
          // last-ditch attempt to salvage the solution by rescaling it to 
          // vsq = 1.0 - 1e-15
          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0) {
            printf("The velocity is superluminal!\n"
                   "Attempting to adjust...\n");
            Real fac = sqrt((1.0 - 1e-15)/vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }
          Real W = sqrt(1.0 / (1.0 - vsq));
          
          host_w0(m, IVX, k, j, i) = W*vu[0];
          host_w0(m, IVY, k, j, i) = W*vu[1];
          host_w0(m, IVZ, k, j, i) = W*vu[2];

          idx++;
        }
      }
    }
  }

  printf("Host mirrors filled.\n");

  // Cleanup
  elliptica_id_reader_free(idr);

  printf("Elliptica freed.\n");

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  //Kokkos::deep_copy(u_z4c, host_u_z4c);

  printf("Data copied.\n");

  //set the inital Ye
  
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmbp->nmb_thispack - 1;
  


  par_for("pgen_tov1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = w0(m,IDN,k,j,i);
    

    Real f = 0.5;
    if (rho >= 2.0e-8) {
      f=0.05;
    } else {
      f = Yfloor;
    }

    w0(m,nvars,k,j,i) = f;
    
  });


  if (pmbp->pmhd != nullptr) {
    // parse some parameters
    b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
    r0 = pin->GetOrAddReal("problem", "r0", 0.0);
    Real r0_ = r0;
    Real b_norm_ = b_norm;

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmbp->nmb_thispack;

    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1, 
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);

      Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
      Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(x1v, x2f, x3f, r0_, b_norm_);
      a2(m,k,j,i) = A2(x1f, x2v, x3f, r0_, b_norm_);
      a3(m,k,j,i) = 0.0;

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(xl,x2f,x3f, r0_, b_norm_) + A1(xr,x2f,x3f, r0_, b_norm_));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(x1f,xl,x3f, r0_, b_norm_) + A2(x1f,xr,x3f, r0_, b_norm_));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  std::cout << "Interpolation and assignment complete!\n";

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    // Complain about something here, because this is a dynamic GR test.
  } else {
    pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              break;
    }
  }

  return;
}




KOKKOS_INLINE_FUNCTION
static Real A1(Real x1, Real x2, Real x3, Real r0, Real b_norm) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return -x2*b_norm*pow(r0,3)/(pow(r0,3)+pow(r,3));
}


KOKKOS_INLINE_FUNCTION
static Real A2(Real x1, Real x2, Real x3, Real r0, Real b_norm) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return x1*b_norm*pow(r0,3)/(pow(r0,3)+pow(r,3));
}

///source function
void neutrinolightbulb(Mesh* pm, const Real bdt){
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int ie = indcs.ie;
  int je = indcs.je;
  int ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  Real Q = C; //Lve,52
  Real Tnu = T; //Tnue
  Real rhocut = rho_cut;
  //Real rhocut2 = rho_cut2;
  //Real rhocut3 = rho_cut3;
  //Real r0_ = r0;
  //Real b_norm_ = b_norm;
  Real BB = B;  
  Real kappatilde = Kappatilde;
  Real gamma = Gamma;
  Real factor = kappatilde * (1.0 - ((1/3)/(gamma - 1.0)));
  Real m = m_;

  //setting up vector potentials
  //////////////////////////////////////
  /*
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb1+1,ncells3,ncells2,ncells1);
  Kokkos::realloc(a2, nmb1+1,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb1+1,ncells3,ncells2,ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

    par_for("pgen_potential", DevExeSpace(), 0,nmb1,ks,ke+1,js,je+1,is,ie+1, 
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);

      Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
      Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(x1v, x2f, x3f, r0_, b_norm_);
      a2(m,k,j,i) = A2(x1f, x2v, x3f, r0_, b_norm_);
      a3(m,k,j,i) = 0.0;

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(xl,x2f,x3f, r0_, b_norm_) + A1(xr,x2f,x3f, r0_, b_norm_));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(x1f,xl,x3f, r0_, b_norm_) + A2(x1f,xr,x3f, r0_, b_norm_));
      }
    }); */

  ////////////////////////////////////////////////


  std::string block;
  DvceArray5D<Real> u0, w0, bcc0;
  int nvars;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;
    u0 = pmbp->phydro->u0;
    w0 = pmbp->phydro->w0;
    block = std::string("hydro");
  } else if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd;
    u0 = pmbp->pmhd->u0;
    w0 = pmbp->pmhd->w0;
    bcc0 = pmbp->pmhd->bcc0;
    block = std::string("mhd");
  }

  auto &b0_ = pmbp->pmhd->b0;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));


    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

    Real alpha = adm.alpha(m, k, j, i);
    Real beta[3] = {adm.beta_u(m,0,k,j,i), adm.beta_u(m,1,k,j,i), adm.beta_u(m,2,k,j,i)};

    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);

    Real utilde[3] = {w0(m,IVX,k,j,i), w0(m,IVY,k,j,i), w0(m,IVZ,k,j,i)};
    Real uu = Primitive::SquareVector(utilde, g3d);
    Real W = sqrt(1.0 + uu); // Lorentz factor

    Real u_[3] = {g3d[S11]*utilde[0] + g3d[S12]*utilde[1] + g3d[S13]*utilde[2],
                  g3d[S12]*utilde[0] + g3d[S22]*utilde[1] + g3d[S23]*utilde[2],
                  g3d[S13]*utilde[0] + g3d[S23]*utilde[1] + g3d[S33]*utilde[2]};

    Real u[3] = {utilde[0] - W*beta[0]/alpha,
                 utilde[1] - W*beta[1]/alpha,
                 utilde[2] - W*beta[2]/alpha};

    Real Bx = bcc0(m,IBX,k,j,i);
    Real By = bcc0(m,IBY,k,j,i);
    Real Bz = bcc0(m,IBZ,k,j,i);

    Real B[3] =   {Bx/vol, By/vol, Bz/vol};  //this is the cursive B

    Real Bv = g3d[S11]*B[0]*utilde[0] + g3d[S12]*B[0]*utilde[1] + g3d[S13]*B[0]*utilde[2] +
              g3d[S12]*B[1]*utilde[0] + g3d[S22]*B[1]*utilde[1] + g3d[S23]*B[1]*utilde[2] +
              g3d[S13]*B[2]*utilde[0] + g3d[S23]*B[2]*utilde[1] + g3d[S33]*B[2]*utilde[2];

    Real b0 = Bv/alpha;

    Real B_[3] = {g3d[S11]*B[0] + g3d[S12]*B[1] + g3d[S13]*B[2],
                  g3d[S12]*B[0] + g3d[S22]*B[1] + g3d[S23]*B[2],
                  g3d[S13]*B[0] + g3d[S23]*B[1] + g3d[S33]*B[2]};

    Real b[3] = {(B[0] + alpha*b0*u[0])/W, (B[1] + alpha*b0*u[1])/W, (B[2] + alpha*b0*u[2])/W}; 
    Real b_[3] = {(B_[0] + alpha*b0*u_[0])/W, (B_[1] + alpha*b0*u_[1])/W, (B_[2] + alpha*b0*u_[2])/W};

    Real Bsq = Primitive::SquareVector(B, g3d);
    Real bsq = (Bv*Bv + Bsq)/(W*W);

    Real hrho = w0(m,IDN,k,j,i) + w0(m,IEN,k,j,i) + 3.0*(w0(m,IEN,k,j,i)- factor*pow(w0(m,IDN,k,j,i),gamma));

    //compute conserved variables at time t
    Real D = vol*W*w0(m,IDN,k,j,i);
    Real S[3] = {vol*((hrho + bsq)*W*u_[0] - Bv*b_[0]),
                 vol*((hrho + bsq)*W*u_[1] - Bv*b_[1]),
                 vol*((hrho + bsq)*W*u_[2] - Bv*b_[2])};
    Real tau = vol*((hrho + bsq)*W*W -(W*w0(m,IDN,k,j,i))-(w0(m,IEN,k,j,i)+0.5*bsq)-(Bv*Bv));

    Real z; //lightbulb cutoff
    if (w0(m,IDN,k,j,i)>rhocut){
      z = exp(1.0 - w0(m,IDN,k,j,i)/rhocut);
    } else {
      z = 1.0;
    }

    /*
    Real z2; //hydro freezing
    if (w0(m,IDN,k,j,i)>rhocut2){
      z2 = exp(2*(1.0 - w0(m,IDN,k,j,i)/rhocut2));
    } else {
      z2 = 1.0;
    }

    Real z3; //mhd freezing
    if (w0(m,IDN,k,j,i)>rhocut3){
      z3 = exp(2*(1.0 - w0(m,IDN,k,j,i)/rhocut3));
    } else {
      z3 = 1.0;
    }*/

  /*
   Real z2; //hydro freezing
   if (r<rcut2){
      z2 = exp(10*(-1.0 + r/rcut2));
    } else {
      z2 = 1.0;
    }

   Real z3; //magnetic freezing
   if (r<rcut){
      z3 = exp(10*(-1.0 + r/rcut));
    } else {
      z3 = 1.0;
    }*/

    //Let's compute the freezing of magnetic field:
    /*
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0_.x1f(m,k,j,i) += (1-z3)*(((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 - (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3) - b0_.x1f(m,k,j,i));
    b0_.x2f(m,k,j,i) += (1-z3)*(((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 - (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1) - b0_.x2f(m,k,j,i));
    b0_.x3f(m,k,j,i) += (1-z3)*(((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 - (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2) - b0_.x3f(m,k,j,i));

      // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0_.x1f(m,k,j,i+1) += (1-z3)*(((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 - (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3) - b0_.x1f(m,k,j,i+1));
    }
    if (j==je) {
      b0_.x2f(m,k,j+1,i) += (1-z3)*(((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 - (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1) - b0_.x2f(m,k,j+1,i));
    }
    if (k==ke) {
      b0_.x3f(m,k+1,j,i) += (1-z3)*(((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 - (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2) - b0_.x3f(m,k+1,j,i));
    }*/


    // Real p = 0.0;
    Real p = fmax(w0(m,IEN,k,j,i) - (kappatilde * pow(w0(m,IDN,k,j,i), gamma)),0.0); //Thermal pressure


    if (r>0.0){

      r = r*pow((1.0 + m/(2*r)),2);   //areal radius which is needed for lightbulb
      Real lambda1 = 0.0109*(Q/pow(r,2))*((4.58*Tnu)+2.586+(0.438/Tnu))+ ((1.285e10)*pow(p,1.25));
      Real lambda2 = lambda1 + 0.0109*(Q/pow(r,2))*((6.477*Tnu)-2.586+(0.309/Tnu))+ ((1.285e10)*pow(p,1.25));

      //star freezing (if z=1 the no extra source term, if z=0 then source sets to unevolved state)
      /*
      u0(m,IDN,k,j,i) += (1-z2)*(D    - u0(m,IDN,k,j,i));
      u0(m,IM1,k,j,i) += (1-z2)*(S[0] - u0(m,IM1,k,j,i));
      u0(m,IM2,k,j,i) += (1-z2)*(S[1] - u0(m,IM2,k,j,i));
      u0(m,IM3,k,j,i) += (1-z2)*(S[2] - u0(m,IM3,k,j,i));
      u0(m,IEN,k,j,i) += (1-z2)*(tau  - u0(m,IEN,k,j,i));*/


      //neutrino lightbulb
      
      u0(m,IM1,k,j,i) += alpha*vol*bdt*w0(m,IDN,k,j,i)*u_[0]*((0.0079*Q*(pow((Tnu/4.0),2)/pow(r,2))) - BB*pow(p,1.5))*z;
      u0(m,IM2,k,j,i) += alpha*vol*bdt*w0(m,IDN,k,j,i)*u_[1]*((0.0079*Q*(pow((Tnu/4.0),2)/pow(r,2))) - BB*pow(p,1.5))*z;
      u0(m,IM3,k,j,i) += alpha*vol*bdt*w0(m,IDN,k,j,i)*u_[2]*((0.0079*Q*(pow((Tnu/4.0),2)/pow(r,2))) - BB*pow(p,1.5))*z;
      u0(m,IEN,k,j,i) += alpha*vol*bdt*w0(m,IDN,k,j,i)*W    *((0.0079*Q*(pow((Tnu/4.0),2)/pow(r,2))) - BB*pow(p,1.5))*z;
      u0(m,nvars,k,j,i) += alpha*vol*bdt*(D)*(lambda1 - lambda2*w0(m,nvars,k,j,i))*z;
      
    }
  });

  return;


}

// History function
void tovFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int ie = indcs.ie;
  int je = indcs.je;
  int ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  Real kappatilde = Kappatilde;
  Real gamma = Gamma;
  Real factor = kappatilde * (1.0 - ((1/3)/(gamma - 1.0)));

  int nvars;
  DvceArray5D<Real> u0, w0, bcc0;  
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    u0 = pmbp->phydro->u0;
    w0 = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    u0 = pmbp->pmhd->u0;
    w0 = pmbp->pmhd->w0;
    bcc0 = pmbp->pmhd->bcc0;
  }
  
  DvceArray5D<Real> alpha("alpha",nmb1+1,1,n1,n2,n3);
  DvceArray5D<Real> beta("beta",nmb1+1,3,n1,n2,n3);
  DvceArray5D<Real> metric("metric",nmb1+1,6,n1,n2,n3);


  par_for("fixing", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    
    metric(m,0,k,j,i) = adm.g_dd(m,0,0,k,j,i);
    metric(m,1,k,j,i) = adm.g_dd(m,0,1,k,j,i);
    metric(m,2,k,j,i)  = adm.g_dd(m,0,2,k,j,i);
    metric(m,3,k,j,i)  = adm.g_dd(m,1,1,k,j,i);
    metric(m,4,k,j,i)  = adm.g_dd(m,1,2,k,j,i);
    metric(m,5,k,j,i)  = adm.g_dd(m,2,2,k,j,i);
    alpha(m,0,k,j,i) = adm.alpha(m, k, j, i);
    beta(m,0,k,j,i) = adm.beta_u(m,0,k,j,i);
    beta(m,1,k,j,i) = adm.beta_u(m,1,k,j,i);
    beta(m,2,k,j,i) = adm.beta_u(m,2,k,j,i);

  });

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = 4;

  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
    pdata->label[nflux*g+3] = "phi_" + rad_str;
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  DualArray2D<Real> interpolated_metric;  // needed for ADM
  DualArray2D<Real> interpolated_alpha;
  DualArray2D<Real> interpolated_beta;

  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    pdata->hdata[nflux*g+3] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    grids[g]->InterpolateToSphere(3, bcc0);
    Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
    Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
    interpolated_bcc.template modify<DevExeSpace>();
    interpolated_bcc.template sync<HostMemSpace>();

    grids[g]->InterpolateToSphere(1, alpha);
    Kokkos::realloc(interpolated_alpha, grids[g]->nangles, 1);
    Kokkos::deep_copy(interpolated_alpha, grids[g]->interp_vals);
    interpolated_alpha.template modify<DevExeSpace>();
    interpolated_alpha.template sync<HostMemSpace>();

    grids[g]->InterpolateToSphere(3, beta);
    Kokkos::realloc(interpolated_beta, grids[g]->nangles, 3);
    Kokkos::deep_copy(interpolated_beta, grids[g]->interp_vals);
    interpolated_beta.template modify<DevExeSpace>();
    interpolated_beta.template sync<HostMemSpace>();

    grids[g]->InterpolateToSphere(6, metric);
    Kokkos::realloc(interpolated_metric, grids[g]->nangles, 6);
    Kokkos::deep_copy(interpolated_metric, grids[g]->interp_vals);
    interpolated_metric.template modify<DevExeSpace>();
    interpolated_metric.template sync<HostMemSpace>();

    grids[g]->InterpolateToSphere(nvars, w0);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);
      Real &int_bx = interpolated_bcc.h_view(n,IBX);
      Real &int_by = interpolated_bcc.h_view(n,IBY);
      Real &int_bz = interpolated_bcc.h_view(n,IBZ);
      Real &int_alpha = interpolated_alpha.h_view(n,0);
      Real int_beta[3] = {interpolated_beta.h_view(n,0),
                          interpolated_beta.h_view(n,1),
                          interpolated_beta.h_view(n,2)};
      Real g3d[NSPMETRIC] =     {interpolated_metric.h_view(n,0),
                                 interpolated_metric.h_view(n,1),
                                 interpolated_metric.h_view(n,2),
                                 interpolated_metric.h_view(n,3),
                                 interpolated_metric.h_view(n,4),
                                 interpolated_metric.h_view(n,5)};
      

      Real r2 = SQR(r);
      Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                  g3d[S22], g3d[S23], g3d[S33]);
      Real sqrtmdet = sqrt(detg);
      Real utilde[3] = {int_vx, int_vy, int_vz};
      Real uu = Primitive::SquareVector(utilde, g3d);
      Real W = sqrt(1.0 + uu); // Lorentz factor
      Real u[3] = {utilde[0] - W*int_beta[0]/int_alpha,
                   utilde[1] - W*int_beta[1]/int_alpha,
                   utilde[2] - W*int_beta[2]/int_alpha};
      Real ur = u[0]*sin(theta)*cos(phi) + u[1]*sin(theta)*sin(phi) + u[2]*cos(theta);
      Real u_[3] = {g3d[S11]*utilde[0] + g3d[S12]*utilde[1] + g3d[S13]*utilde[2],
                    g3d[S12]*utilde[0] + g3d[S22]*utilde[1] + g3d[S23]*utilde[2],
                    g3d[S13]*utilde[0] + g3d[S23]*utilde[1] + g3d[S33]*utilde[2]};
      Real u_phi = (-u_[0]*sin(phi) + u_[1]*cos(phi))*r*sin(theta);
      Real hrho = int_dn + int_ie + 3.0*(int_ie - factor*pow(int_dn,gamma));


      Real B[3] = {int_bx/sqrtmdet,
                   int_by/sqrtmdet,
                   int_bz/sqrtmdet};  //this is the cursive B

      Real Bv = g3d[S11]*B[0]*utilde[0] + g3d[S12]*B[0]*utilde[1] + g3d[S13]*B[0]*utilde[2] +
                g3d[S12]*B[1]*utilde[0] + g3d[S22]*B[1]*utilde[1] + g3d[S23]*B[1]*utilde[2] +
                g3d[S13]*B[2]*utilde[0] + g3d[S23]*B[2]*utilde[1] + g3d[S33]*B[2]*utilde[2];

      Real b0 = Bv/int_alpha;
      Real B_[3] = {g3d[S11]*B[0] + g3d[S12]*B[1] + g3d[S13]*B[2],
                    g3d[S12]*B[0] + g3d[S22]*B[1] + g3d[S23]*B[2],
                    g3d[S13]*B[0] + g3d[S23]*B[1] + g3d[S33]*B[2]};

      Real b[3] = {(B[0] + int_alpha*b0*u[0])/W, (B[1] + int_alpha*b0*u[1])/W, (B[2] + int_alpha*b0*u[2])/W}; 
      Real b_[3] = {(B_[0] + int_alpha*b0*u_[0])/W, (B_[1] + int_alpha*b0*u_[1])/W, (B_[2] + int_alpha*b0*u_[2])/W};
      Real br = b[0]*sin(theta)*cos(phi) + b[1]*sin(theta)*sin(phi) + b[2]*cos(theta);
      Real b_phi = (-b_[0]*sin(phi) + b_[1]*cos(phi))*r*sin(theta);

      Real Bsq = Primitive::SquareVector(B, g3d);
      Real bsq = (Bv*Bv + Bsq)/(W*W);

      Real betau = g3d[S11]*int_beta[0]*u[0] + g3d[S12]*int_beta[0]*u[1] + g3d[S13]*int_beta[0]*u[2] +
                   g3d[S12]*int_beta[1]*u[0] + g3d[S22]*int_beta[1]*u[1] + g3d[S23]*int_beta[1]*u[2] +
                   g3d[S13]*int_beta[2]*u[0] + g3d[S23]*int_beta[2]*u[1] + g3d[S33]*int_beta[2]*u[2];

      Real betab = g3d[S11]*int_beta[0]*b[0] + g3d[S12]*int_beta[0]*b[1] + g3d[S13]*int_beta[0]*b[2] +
                   g3d[S12]*int_beta[1]*b[0] + g3d[S22]*int_beta[1]*b[1] + g3d[S23]*int_beta[1]*b[2] +
                   g3d[S13]*int_beta[2]*b[0] + g3d[S23]*int_beta[2]*b[1] + g3d[S33]*int_beta[2]*b[2];

      Real u_0 = - int_alpha*W + betau;
      Real u0 =  W/int_alpha;
      Real b_0 = - int_alpha*int_alpha*b0 + betab;
      Real Br = B[0]*sin(theta)*cos(phi) + B[1]*sin(theta)*sin(phi) + B[2]*cos(theta);

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      

      // compute mass flux
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*r2*domega*sqrtmdet*int_alpha;

     // compute energy flux
      Real t1_0 = (hrho + bsq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega*int_alpha*r2;

      // compute angular momentum flux
      Real t1_3 = (hrho + bsq)*ur*u_phi - br*b_phi;
      pdata->hdata[nflux*g+2] += -1.0*t1_3*sqrtmdet*domega*int_alpha*r2;

      // compute magnetic flux
      pdata->hdata[nflux*g+3] += 0.5*fabs(Br)*sqrtmdet*domega*int_alpha*r2;
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
 }

























/*
// Boundary function
void VacuumBC(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  
  DvceArray5D<Real> u0_, w0_;
  u0_ = pm->pmb_pack->pmhd->u0;
  w0_ = pm->pmb_pack->pmhd->w0;
  auto &b0 = pm->pmb_pack->pmhd->b0;
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  Real &dfloor = pm->pmb_pack->pmhd->peos->eos_data.dfloor;
  
  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of the computational domain.
  par_for("noinflow_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
        b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
        if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
        b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
        if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        u0_(m, IDN, k, j, is-i-1) = dfloor;
        u0_(m, IM1, k, j, is-i-1) = 0.0;
        u0_(m, IM2, k, j, is-i-1) = 0.0;
        u0_(m, IM3, k, j, is-i-1) = 0.0;
        u0_(m, IEN, k, j, is-i-1) = 0.0;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
        b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
        if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
        b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
        if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        u0_(m, IDN, k, j, ie+i+1) = dfloor;
        u0_(m, IM1, k, j, ie+i+1) = 0.0;
        u0_(m, IM2, k, j, ie+i+1) = 0.0;
        u0_(m, IM3, k, j, ie+i+1) = 0.0;
        u0_(m, IEN, k, j, ie+i+1) = 0.0;
      }
    }
  });

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
        if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
        b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
        b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
        if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        u0_(m, IDN, k, js-j-1, i) = dfloor;
        u0_(m, IM1, k, js-j-1, i) = 0.0;
        u0_(m, IM2, k, js-j-1, i) = 0.0;
        u0_(m, IM3, k, js-j-1, i) = 0.0;
        u0_(m, IEN, k, js-j-1, i) = 0.0;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
        if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
        b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
        b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
        if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        u0_(m, IDN, k, je+j+1, i) = dfloor;
        u0_(m, IM1, k, je+j+1, i) = 0.0;
        u0_(m, IM2, k, je+j+1, i) = 0.0;
        u0_(m, IM3, k, je+j+1, i) = 0.0;
        u0_(m, IEN, k, je+j+1, i) = 0.0;
      }
    }
  });

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
        if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
        b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
        if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
        b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        u0_(m, IDN, ks-k-1, j, i) = dfloor;
        u0_(m, IM1, ks-k-1, j, i) = 0.0;
        u0_(m, IM2, ks-k-1, j, i) = 0.0;
        u0_(m, IM3, ks-k-1, j, i) = 0.0;
        u0_(m, IEN, ks-k-1, j, i) = 0.0;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
        if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
        b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
        if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
        b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        u0_(m, IDN, ke+k+1, j, i) = dfloor;
        u0_(m, IM1, ke+k+1, j, i) = 0.0;
        u0_(m, IM2, ke+k+1, j, i) = 0.0;
        u0_(m, IM3, ke+k+1, j, i) = 0.0;
        u0_(m, IEN, ke+k+1, j, i) = 0.0;
      }
    }
  });
}

// History function
void TOVHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  int &nmhd = pm->pmb_pack->pmhd->nmhd;
  pdata->nhist = 1;
  pdata->label[0] = "rho-max";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max;
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
  }, Kokkos::Max<Real>(rho_max));

  // store data in hdata array
  pdata->hdata[0] = rho_max;
}
*/