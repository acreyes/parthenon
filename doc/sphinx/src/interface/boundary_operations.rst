.. This file was made in part with generative AI.

.. _boundary-ops:

Registered Boundary Operations
==============================

By default, a ``FillGhost`` variable is exchanged by copying a logical slab of
the sender's cells into a buffer (``SendBoundBufs``) and copying each received
buffer into the receiver's ghost zones (``SetBounds``), one buffer per
neighbor. A variable can replace either copy with its own operation, in the
same way it can register prolongation and restriction operations (see
:doc:`refinement_operations`). This is meant
for exchanges the per-buffer copy cannot express, for example a ghost region
that several neighbors map onto (a node of valence other than four in a 2D
forest) or a ghost fill that combines data from several buffers.

Registering
-----------

.. code:: c++

   Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
   m.RegisterBoundaryOps<MyUnpack>();           // custom unpack, default pack
   m.RegisterBoundaryOps<MyUnpack, MyPack>();   // custom unpack and pack
   m.RegisterBoundaryOps<parthenon::boundary_ops::DefaultUnpack, MyPack>();

The variable must be ``FillGhost``. ``boundary_ops::DefaultPack`` and
``boundary_ops::DefaultUnpack`` keep the default copy on that side. Each op is
a struct with one static function:

.. code:: c++

   struct MyUnpack {
     static void Unpack(MeshData<Real> *md, const BndInfoArr_t &info,
                        const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                        const BndOpIdxHost_t &idx_h, std::size_t nbuffers);
   };
   struct MyPack {
     static void Pack(/* same arguments */);
   };

The ops are hashed by their types (like refinement ops), so all variables that
register the same pair share one op set.

Calling sequence
----------------

The boundaries of a cache (one ``MeshData`` and one ``BoundaryType``) are
bucketed by op set. ``SetBounds`` runs the default unpack kernel over the
boundaries without an unpack op, then calls each registered ``Unpack`` once
with its whole bucket; ``SendBoundBufs`` does the same with ``Pack`` before
the buffers are sent (after restriction, if any). Because an op sees every
buffer of its bucket at once, a single kernel can combine several neighbors'
buffers.

- ``info`` and ``info_h`` are the cache's ``BndInfo`` array on device and
  host; ``idx(n)``, ``0 <= n < nbuffers``, are the entries of the bucket.
  Each ``BndInfo`` has the usual fields used by the default kernels
  (``idxer``, ``lcoord_trans``, ``topo_idx``, ``var``, ``buf``,
  ``allocated``, ``buf_allocated``) and, for the ops:

  - ``block``: index in ``md`` of the local block (the sender in ``Pack``,
    the receiver in ``Unpack``),
  - ``send_gid``, ``recv_gid``: the gids of the two blocks,
  - ``offsets``: the neighbor's offset seen from the local block.

- **Order.** The bucket is sorted by (``block``, variable label, neighbor
  offset index, neighbor gid), where the neighbor gid is ``recv_gid`` in
  ``Pack`` and ``send_gid`` in ``Unpack``. The order depends only on the
  mesh, not on the number of ranks or the order in which buffers arrive, so
  an op that combines buffers in bucket order gives bitwise identical results
  on any number of ranks.
- A buffer filled by ``Pack`` uses the default buffer size and is always sent
  (it is never sent as a null buffer of a sparse variable).
- An op runs on host and launches its own kernels in ``DevExecSpace()``. It
  should write only its own variables (``Unpack``) or buffers (``Pack``).
- Ops act on the variable's own ghost exchange for every ``BoundaryType`` it
  takes part in (including GMG exchanges), not on flux correction (see
  `Flux-correction combine ops`_).
- Variables without registered ops are not affected: if no variable in a
  cache has ops, the cache holds no op buckets and the default kernels run
  over all boundaries exactly as before.

Example
-------

``example/boundary_ops`` builds a 2D forest of fans whose central nodes have
valence 3 to 6. Its variable registers ``AverageOverlappingGhosts``, which
fills each ghost region with the mean of all buffers that map onto it, summed
in ascending sender gid:

.. code:: c++

   // Consecutive entries with the same (block, offsets) fill one ghost region.
   Kokkos::parallel_for(
       "AverageOverlappingGhosts",
       Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nregions, Kokkos::AUTO),
       KOKKOS_LAMBDA(parthenon::team_mbr_t team) {
         const int first = region_start(team.league_rank());
         const int last = region_start(team.league_rank() + 1);
         for (int n = first; n < last; ++n) {
           auto &bi = info(idx(n));
           // ... map buffer element m to ghost cell (kl, jl, il) as the default
           // unpack does, then
           ghost = (n == first) ? bi.buf(m) : ghost + bi.buf(m);
           if (n == last - 1) ghost /= count;
           team.team_barrier();
         }
       });

At valence 5 and 6 two or three node neighbors share a corner ghost region of
the blocks at the central node; with the default copy the region would hold
whichever buffer was unpacked last. The regression test ``boundary_ops`` runs
the example on 1, 2 and 4 ranks, checks every ghost region against the
neighbor average and requires the ghost zones of all runs to be bitwise
identical.

Flux-correction combine ops
---------------------------

Flux correction normally runs only between refinement levels: the fine block
restricts its flux on the shared elements and the coarse block overwrites its
own flux with it. A flux variable can instead (or in addition) reconcile its
fluxes with neighbors **on the same level**, where each block computed its own
candidate flux on the shared elements and the copies are combined, for
example averaged, so that every block holds the same value:

.. code:: c++

   Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes});
   m.RegisterFluxCorrectionOps<MyCombine>();   // registered on the flux variable

   struct MyCombine {
     static void Combine(MeshData<Real> *md, const BndInfoArr_t &info,
                         const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                         const BndOpIdxHost_t &idx_h, std::size_t nbuffers);
   };

Called on a variable with ``WithFluxes``, the op is registered on its flux
variable; it may also be called on a variable that is itself a ``Flux``. Ops
are hashed by type and enumerated in the ``StateDescriptor``
(``FluxCorrectionFuncID``) like the boundary ops above.

- **Admission.** The flux-correction exchange (``flxcor_send`` /
  ``flxcor_recv``) includes same-level neighbors only for flux variables with
  a combine op. They take the same shared elements as across levels: the
  shared face of a face flux (from a cell variable); the shared faces and
  edges of an edge flux (from a face variable; in 2D the edges along
  :math:`x_3` are the nodes of the mesh); the shared faces, edges and nodes of
  a node flux (from an edge variable). Each element shared by :math:`n`
  blocks on the same level therefore arrives in :math:`n - 1` buffers, one
  from every other block sharing it, so an op can combine all :math:`n`
  values, for example two at a face or the valence at a node of a 2D forest.
- **Call.** ``SetBounds<BoundaryType::flxcor_recv>`` first runs the default
  overwrite over the cross-level boundaries, then calls each ``Combine`` once
  with its bucket of same-level boundaries. ``var`` of each ``BndInfo`` is the
  local flux, ``buf`` the neighbor's flux on the shared elements, in the
  neighbor's frame (map the indices and the sign as the default unpack does).
  The bucket is sorted as above, by (``block``, variable label, offset index,
  ``send_gid``). Boundaries to other levels keep the default restriction and
  overwrite, also for variables with a combine op.
- **Tasks.** ``AddFluxCorrectionTasks`` adds the tasks if the mesh is
  multilevel or any variable registered a combine op
  (``HasFluxCorrectionOps(md)``), and then makes ``SetBounds`` wait for the
  send of the same ``MeshData``, since a combine op overwrites fluxes that the
  send must have packed first. Codes that add the flux-correction tasks
  themselves must do the same.
- **Order.** IEEE addition commutes, so the mean of two values is bitwise
  identical on both sides; a sum of more values is identical on every block
  only if every block adds them in the same order, e.g. by ascending gid
  (``send_gid`` of the buffers and ``recv_gid`` of the local value). Values
  in frames with opposite signs (``fac = -1``) give exactly negated sums.
- Variables without a combine op are not affected: they exchange no
  same-level flux buffers, and if no variable has a combine op the
  flux-correction caches hold no buckets and run the default kernels as
  before. Sparse pools do not carry combine ops yet.

The ``boundary_ops`` example registers ``AverageSharedFluxes`` for the face
flux of a cell variable and the edge flux of a face variable. Every block sets
its fluxes to block-dependent values; after the exchange every shared element
holds the mean over all blocks sharing it, summed in ascending gid: two at
faces and the nodes along edges, and three to six at the central nodes of the
fans. The op runs one team per buffer; the first buffer of a (block, variable)
group that holds an element collects the local value and the value of every
buffer of the group holding it, sorts them by gid and writes the mean. The
example checks every shared element on host, and the regression test
requires the fluxes of the 1, 2 and 4 rank runs to be bitwise identical.
