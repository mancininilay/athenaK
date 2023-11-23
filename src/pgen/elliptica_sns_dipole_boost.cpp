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
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyngr/dyngr.hpp"
#include "elliptica_id_reader_lib.h"

#if ELLIPTICA==0
#error elliptica_bns.cpp requires Elliptica
#endif



// Prototypes for functions used internally in this pgen.


KOKKOS_INLINE_FUNCTION
static Real A1(Real b_norm, Real r0, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
static Real A2(Real b_norm, Real r0, Real x1, Real x2, Real x3);

// Prototypes for user-defined BCs and history
void TOVHistory(HistoryData *pdata, Mesh *pm);
void VacuumBC(Mesh *pm);

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

  // FIXME: Set boundary condition function?
  //user_bcs_func = VacuumBC;

  Real kappa;
  Real gamma;
  Real dfloor;
  Real pfloor;


  // Select either Hydro or MHD
  std::string block;
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
 //   u0_ = pmbp->phydro->u0;
 //   w0_ = pmbp->phydro->w0;
    block = std::string("hydro");
  } else if (pmbp->pmhd != nullptr) {
  //  u0_ = pmbp->pmhd->u0;
  //  w0_ = pmbp->pmhd->w0;
    block = std::string("mhd");
  }

  kappa = pin->GetReal("problem", "kappa");
  gamma = pin->GetOrAddReal(block, "gamma", 5.0/3.0);
  dfloor = pin->GetOrAddReal(block, "dfloor", (FLT_MIN));
  pfloor = pin->GetOrAddReal(block, "pfloor", (FLT_MIN));


  if (pmbp->pdyngr->eos_policy != DynGR_EOS::eos_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sns star problem currently only compatible with eos_ideal"
              << std::endl;
  }


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



  if (pmbp->pmhd != nullptr) {
    // parse some parameters
    Real b_norm;
    Real r0;
    b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
    r0 = pin->GetOrAddReal("problem", "r0", 0.0);


    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmbp->nmb_thispack;
    DvceArray4D<Real> a1, a2, a3, a1p, a2p, a3p;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a1p, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2p, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3p, nmb,ncells3,ncells2,ncells1);

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

      auto w0c   = pmbp->pmhd->w0;
      Real wx = w0c(m,IVX,k,j,i);
      Real wy = w0c(m,IVY,k,j,i);
      Real wz = w0c(m,IVZ,k,j,i);
      Real ww = (wx*wx + wy*wy + wz*wz);
      Real vv = ww / (1.0 + ww);
      Real lorentz = 1/sqrt(1-vv);
      Real vx = wx/lorentz;
      Real vy = wy/lorentz;
      Real vz = wz/lorentz;

      a1p(m,k,j,i) = A1(b_norm,r0, x1v, x2f, x3f);
      a2p(m,k,j,i) = A2(b_norm,r0, x1f, x2v, x3f);
      a3p(m,k,j,i) = 0.0;
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
        a1p(m,k,j,i) = 0.5*(A1(b_norm,r0, xl,x2f,x3f) + A1(b_norm,r0, xr,x2f,x3f));
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
        a2p(m,k,j,i) = 0.5*(A2(b_norm, r0,x1f,xl,x3f) + A2(b_norm,r0, x1f,xr,x3f));
      }

      //implement lorentz transformation
      if (vx > 0 && vy > 0) {
        a1(m,k,j,i) = (1+((lorentz-1)*(vx*vx)/(vv)))*a1p(m,k,j,i) + ((lorentz-1)*(vx*vy)*a2p(m,k,j,i))/(vv);
        a2(m,k,j,i) = (1+((lorentz-1)*(vy*vy)/(vv)))*a2p(m,k,j,i) + ((lorentz-1)*(vx*vy)*a1p(m,k,j,i))/(vv);
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
    pmbp->pdyngr->PrimToConInit(0, (ncells1-1), 0, (ncells2-1), 0, (ncells3-1));
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
static Real A1(Real b_norm, Real r0, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return -x2*b_norm*pow(r0,3)/(pow(r0,3)+pow(r,3));
}


KOKKOS_INLINE_FUNCTION
static Real A2(Real b_norm, Real r0, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return x1*b_norm*pow(r0,3)/(pow(r0,3)+pow(r,3));
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