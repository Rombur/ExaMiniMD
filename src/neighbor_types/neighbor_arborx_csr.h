//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

// Runtime Check for this Neighbor Class
#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "ARBORX_CSR") == 0) )
        neighbor_type = NEIGH_ARBORX_CSR;
#endif

// Instantiation and Init of this class
#ifdef NEIGHBOR_MODULES_INSTANTIATION
    else if (input->neighbor_type == NEIGH_ARBORX_CSR) {
      neighbor = new NeighborArborXCSR<t_neigh_mem_space>();
      neighbor->init(input->force_cutoff + input->neighbor_skin);
    }
#endif

// Add Force Instantiation case
#if defined(FORCE_MODULES_INSTANTIATION)
      case NEIGH_ARBORX_CSR: force = new FORCETYPE_ALLOCATION_MACRO(NeighborArborXCSR<t_neigh_mem_space>); break;
#endif

// Add Force declaration line
#if defined(FORCE_MODULES_EXTERNAL_TEMPLATE)
      extern template class FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighborArborXCSR<t_neigh_mem_space>);
#endif

// Add Force Template Instantiation line
#if defined(FORCE_MODULES_TEMPLATE)
      template class FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighborArborXCSR<t_neigh_mem_space>);
#endif


// Making sure we are not just instantiating some Option
#if !defined(MODULES_OPTION_CHECK) && \
    !defined(NEIGHBOR_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_EXTERNAL_TEMPLATE) && \
    !defined(FORCE_MODULES_TEMPLATE)
#include <neighbor.h>
#ifndef NEIGHBOR_ARBORX_CSR_H
#define NEIGHBOR_ARBORX_CSR_H
#include <ArborX.hpp>
#include <system.h>
#include <binning.h>
#include <neighbor_list_csr.h>

template <typename MemorySpace>
class NeighborArborXCSR : public Neighbor
{
  protected:
    T_X_FLOAT neigh_cut = 0.0;
    t_x x;
    t_type type;
    t_id id;

    T_INT nbinx, nbiny, nbinz, nhalo;
    T_INT N_local;

    Kokkos::View<T_INT*, MemorySpace> num_neighs;
    Kokkos::View<T_INT*, MemorySpace> neigh_offsets;
    bool half_neigh;

    typename Binning::t_binoffsets bin_offsets;
    typename Binning::t_bincount bin_count;
    typename Binning::t_permute_vector permute_vector;

  public:
    typedef NeighListCSR<MemorySpace> t_neigh_list;

    t_neigh_list neigh_list;

    NeighborArborXCSR() = default;

    ~NeighborArborXCSR() = default;

    void init(T_X_FLOAT neigh_cut_) override
    {
      neigh_cut = neigh_cut_;
    }

    void create_neigh_list(System* system, Binning* binning, bool half_neigh_, 
                           bool /*ghost_neighs_*/) override
    {
      // Get some data handles
      N_local = system->N_local;
      x = system->x;
      type = system->type;
      id = system->id;
      half_neigh = half_neigh_;
      unsigned int const n_particles =  system->N_local+system->N_ghost;

			Kokkos::View<int *, MemorySpace> offset("offset", 0);
			Kokkos::View<int *, MemorySpace> indices("indices", 0);

      if(half_neigh)
      {
        // Half neighbor is not implemented yet
        throw std::runtime_error("Not implemented");
      }
      else
      {
        using ExecutionSpace = Kokkos::DefaultExecutionSpace;

        // Spatial search. Radius is neigh_cut
        t_x x_dummy = x;
        Kokkos::View<ArborX::Point *, MemorySpace> particles(
          Kokkos::ViewAllocateWithoutInitializing("particles"), n_particles);
        Kokkos::parallel_for(
          "fill_particles",
          Kokkos::RangePolicy<ExecutionSpace>(0, n_particles), KOKKOS_LAMBDA(int i) {
            for (int d = 0; d<3; ++d)
            particles(i)[d] = x_dummy(i,d);
          });

        T_X_FLOAT neigh_cut_dummy = neigh_cut;
        Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, MemorySpace>
          queries(Kokkos::ViewAllocateWithoutInitializing("queries"), N_local);
        Kokkos::parallel_for(
          "setup_radius_search_queries",
          Kokkos::RangePolicy<ExecutionSpace>(0, N_local), KOKKOS_LAMBDA(int i) {
            queries(i) = ArborX::intersects(ArborX::Sphere{{static_cast<float>(x_dummy(i,0)), 
              static_cast<float>(x_dummy(i,1)), static_cast<float>(x_dummy(i,2))}, 
              neigh_cut_dummy});
          });

        // Perform the Search
			  Kokkos::View<int *, MemorySpace> tmp_indices("tmp_indices", 0);
				ArborX::BVH<MemorySpace> bvh(ExecutionSpace{}, particles);
				bvh.query(ExecutionSpace{}, queries, tmp_indices, offset);

        Kokkos::realloc(indices, tmp_indices.extent(0)-N_local);
        int n = 0;
        for (unsigned int i=0; i<static_cast<int>(offset.extent(0))-1; ++i)
        { 
          for (unsigned int j=offset(i); j<offset(i+1); ++j)
          {
            if (tmp_indices(j) != i)
            {
              indices(n) = tmp_indices(j);
              ++n;
            }
          }
        }
        for (unsigned int i=1; i<offset.extent(0); ++i)
          offset(i) -= i;
      }

      // Create actual CSR NeighList
      neigh_list = t_neigh_list(indices, offset);
    }

    t_neigh_list get_neigh_list()
    {
      return neigh_list;
    }

    const char* name() override
    {
      return "NeighborArborXCSR";
    }

};

template <>
struct NeighborAdaptor<NEIGH_ARBORX_CSR>
{
  typedef NeighborArborXCSR<t_neigh_mem_space> type;
};

extern template class NeighborArborXCSR<t_neigh_mem_space>;
#endif
#endif
