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
Members of a sparse pool inherit the ops registered on the pool's metadata.

Same-level only
---------------

Registered ops handle the exchanges between blocks on the **same refinement
level** only. Boundaries to a coarser or finer neighbor always take
Parthenon's default path: restriction into the send buffer, the default pack
and unpack, and prolongation of the received coarse data, using the
variable's registered prolongation and restriction operations (see
:doc:`refinement_operations`). Customise cross-level ghost fills with those
operations, not with boundary ops. An op therefore never sees a coarse
buffer, and every buffer in its bucket has the shape of a same-level ghost
region.

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
- Ops act on the variable's own same-level ghost exchange for every
  ``BoundaryType`` it takes part in (including GMG exchanges), not on flux
  correction.
- Variables without registered ops are not affected: if no variable in a
  cache has ops, the cache holds no op buckets and the default kernels run
  over all boundaries exactly as before.

Library ops
-----------

``bvals/comms/bnd_ops_library.hpp`` provides ready-made ops in
``parthenon::boundary_ops``. They launch device kernels and are deterministic:
their result is bitwise independent of the number of ranks and of the order in
which buffers arrive.

- ``AverageOverlapping`` (unpack): every ghost element gets the mean of all
  same-level buffers that land on it, summed in ascending sender gid. On a
  regular mesh each ghost region has one sender and this is the default copy.
  Where several neighbors map onto one region, e.g. the node neighbors of a
  node of valence other than four in a 2D forest, the default copy would keep
  whichever buffer is unpacked last. Consecutive bucket entries with the same
  (``block``, variable, ``offsets``) are one region; one team per region
  zeroes it, adds the buffers one at a time in bucket order and divides by
  their number. Unallocated (sparse) sender buffers count as the variable's
  sparse default value.

.. code:: c++

   #include <bvals/comms/bnd_ops_library.hpp>

   m.RegisterBoundaryOps<parthenon::boundary_ops::AverageOverlapping>();

Example
-------

``example/boundary_ops`` builds a 2D forest of fans whose central nodes have
valence 3 to 6. A dense variable and a member of a sparse pool both register
``AverageOverlapping``. At valence 5 and 6 two or three node neighbors share a
corner ghost region of the blocks at the central node; with the default copy
the region would hold whichever buffer was unpacked last. The example checks
every ghost region of both variables against the neighbor average. The
regression test ``boundary_ops`` runs it on 1, 2 and 4 ranks and requires the
ghost zones of all runs to be bitwise identical.
