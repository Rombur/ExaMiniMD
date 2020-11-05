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

#ifndef NEIGHBOR_LIST_CSR_H
#define NEIGHBOR_LIST_CSR_H

#include <Kokkos_StaticCrsGraph.hpp>

#include <iostream>
#include <set>

template<class MemorySpace>
struct NeighListCSR : public Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,void,T_INT> {
  struct NeighViewCSR {
    private:
      const T_INT* const ptr;
      const T_INT num_neighs;

    public:
      KOKKOS_INLINE_FUNCTION
      NeighViewCSR (const T_INT* const ptr_, const T_INT& num_neighs_):
        ptr(ptr_),num_neighs(num_neighs_) {}

      KOKKOS_INLINE_FUNCTION
      T_INT operator() (const T_INT& i) const { return ptr[i]; }

      KOKKOS_INLINE_FUNCTION
      T_INT get_num_neighs() const { return num_neighs; }
  };

  typedef NeighViewCSR t_neighs;

  NeighListCSR() :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,void,T_INT>() {}
  NeighListCSR (const NeighListCSR& rhs) :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,void,T_INT>(rhs) {
  }

  template<class EntriesType, class RowMapType>
  NeighListCSR (const EntriesType& entries_,const RowMapType& row_map_) :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,void,T_INT>( entries_, row_map_) {}


  KOKKOS_INLINE_FUNCTION
  T_INT get_num_neighs(const T_INT& i) const {
    return this->row_map(i+1) - this->row_map(i);
  }

  KOKKOS_INLINE_FUNCTION
  t_neighs get_neighs(const T_INT& i) const {
    const T_INT start = this->row_map(i);
    const T_INT end = this->row_map(i+1);
    return t_neighs(&this->entries(start),end-start);
  }
};

#endif
