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
  (it is never sent as a null buffer of a sparse variable). A variable-length
  ``Pack`` sends only a prefix of it, see below.
- An op runs on host and launches its own kernels in ``DevExecSpace()``. It
  should write only its own variables (``Unpack``) or buffers (``Pack``).
- Ops act on the variable's own same-level ghost exchange for every
  ``BoundaryType`` it takes part in (including GMG exchanges), not on flux
  correction (see `Flux-correction combine ops`_).
- Variables without registered ops are not affected: if no variable in a
  cache has ops, the cache holds no op buckets and the default kernels run
  over all boundaries exactly as before.

Variable-length payloads
------------------------

A buffer's allocated size is its **capacity**; by default the whole buffer is
sent. An op whose data volume changes from step to step (e.g. a number of
neighbor rows that depends on a moving geometry) can send only the leading
**payload** of each buffer. It declares this by taking two more arguments, the
payload lengths in buffer elements on device and host, indexed like the
``BndInfo`` array:

.. code:: c++

   struct MyPack {
     static void Pack(MeshData<Real> *md, const BndInfoArr_t &info,
                      const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                      const BndOpIdxHost_t &idx_h, std::size_t nbuffers,
                      const BndPayload_t &payload, const BndPayloadHost_t &payload_h);
   };
   struct MyUnpack {
     static void Unpack(/* same arguments */);
   };

- ``Pack``: before the call ``payload_h(idx_h(n))`` holds the capacity
  (``info_h(idx_h(n)).buf.size()``). The op fills the first ``len`` elements
  of the buffer and sets ``payload_h(idx_h(n)) = len`` on host (if it computes
  the lengths on device, it copies them to ``payload_h`` itself; ``payload`` is
  not read). Only those ``len`` elements are sent. ``len`` must be in
  ``[1, capacity]``; anything else is an error (``SendBoundBufs`` throws). A
  zero-length message would mean a null buffer of a sparse variable, so send
  at least one element.
- ``Unpack``: ``payload_h(idx_h(n))`` and ``payload(idx(n))`` hold the
  number of elements received. Elements past it are stale (left from earlier
  messages, or, for same-rank buffers, whatever the sender wrote there).
- Each side opts in separately: an ``Unpack`` that takes the lengths gets them
  even if its ``Pack`` sends whole buffers; a variable-length ``Pack`` must be
  paired with a registered ``Unpack``. Ops that do not take the lengths are
  called as before.
- How the length gets to the receiver: ``CommBuffer::Send(size)`` posts an
  MPI message of ``size`` elements, while receives are posted with the full
  capacity, so MPI accepts the shorter message; the receiving ``CommBuffer``
  stores the count of the completed receive (``MPI_Get_count``, which it
  already called to detect null buffers). A same-rank buffer is one object on
  both ends and stores the length given to ``Send``. Both are read with
  ``CommBuffer::GetPayloadSize()``.
- The capacity is fixed when the buffers are built. To send more than the
  capacity, allocate larger buffers and rebuild them (for provider lanes:
  return a larger ``BufferSize`` and call ``Mesh::RebuildLane``).
- Variable-length ops cannot be combined with coalesced communication
  (``parthenon/mesh/do_coalesced_comms``).
- Cost: buffers without a variable-length op are sent as before (whole
  buffer). The bookkeeping is one integer store per ``Send`` and per completed
  MPI receive, plus, for caches with registered ops only, a host loop over
  the variable-length buckets and one small host-to-device copy before their
  ``Unpack``.

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
every ghost region of both variables against the neighbor average. A third
variable, ``varlen``, registers a variable-length pack op that sends
``1 + (7 send_gid + 3 recv_gid + 5 cycle) mod capacity`` elements per buffer
over three cycles, and an unpack op that checks the received lengths and
values. The
regression test ``boundary_ops`` runs it on 1, 2 and 4 ranks and requires the
ghost zones of all runs to be bitwise identical.

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
  before. Members of a sparse pool inherit the combine op of the pool's
  metadata.

Library op ``flux_correction_ops::AverageShared`` (in
``bvals/comms/bnd_ops_library.hpp``): every shared element gets the mean of
the local value and the values of all same-level neighbors sharing it, summed
in ascending gid, so every block holds the same bits (up to the sign of its
frame) on any number of ranks. It runs one team per buffer; the first buffer of
a (block, variable) group that holds an element collects the local value and
the value of every buffer of the group holding it, sorts them by gid and
writes the mean. Up to ``AverageShared::kMaxShared`` (16) blocks may share an
element.

.. code:: c++

   #include <bvals/comms/bnd_ops_library.hpp>

   m.RegisterFluxCorrectionOps<parthenon::flux_correction_ops::AverageShared>();

The ``boundary_ops`` example registers ``AverageShared`` for the face flux of
a cell variable, the edge flux of a face variable and the face flux of a
sparse-pool member. Every block sets its fluxes to block-dependent values;
after the exchange every shared element holds the mean over all blocks sharing
it: two at faces and the nodes along edges, and three to six at the central
nodes of the fans. The example checks every shared element on host, and the
regression test requires the fluxes of the 1, 2 and 4 rank runs to be bitwise
identical.

Provider lanes
--------------

A provider lane exchanges variables between blocks that a user-registered
neighbor provider declares to be neighbors, instead of the blocks' geometric
neighbors. The motivating case is a pool of blocks on a hyper-rectangular mesh
whose real connectivity (unstructured, with nodes of any valence, changing in
time) is known only to the downstream code. Parthenon fills in ranks, builds
tag maps and buffers, and runs the exchanges and their MPI; the registered
boundary ops above (pack, unpack) and flux-correction combine ops run on the
lane buffers.

.. code:: cpp

  constexpr parthenon::LaneId my_lane{0};  // small non-negative ids

  // Package: lane variables, e.g. per-block records
  Metadata m({Metadata::None, Metadata::Derived}, std::vector<int>{4});
  m.SetLane(my_lane);                        // before registering ops
  m.RegisterBoundaryOps<MyUnpack, MyPack>(); // both required on a lane
  Metadata mq({Metadata::Cell, Metadata::WithFluxes});
  mq.SetLane(my_lane);
  mq.RegisterFluxCorrectionOps<MyCombine, MyFluxPack>();

  // After the mesh is built, on every rank
  pmesh->RegisterNeighborProvider(
      my_lane, [](const MeshBlock &pmb, std::vector<LaneNeighbor> &nbs) {
        nbs.push_back({/*gid=*/nb_gid, /*channel_me=*/c, /*channel_nb=*/c_nb});
      });

  // Task list; md may also hold other lanes' and ordinary variables
  auto ex = AddLaneExchangeTasks(dep, tl, md, my_lane);
  auto fx = AddLaneFluxCorrectionTasks(dep, tl, md, my_lane);

  // After the downstream code changed the lane graph, between exchanges
  pmesh->RebuildLane(my_lane);

- **Provider.** Called for every local block, it lists ``{gid, channel_me,
  channel_nb}`` per lane neighbor. Channel ids are user labels in
  ``[0, MaxLaneChannels)`` (e.g. a local face or node index); a block may list
  the same gid several times with different channels, and several neighbors
  on the same channel (e.g. all blocks around a node). The graph must be
  symmetric: if block ``a`` lists ``{b, ca, cb}``, block ``b`` lists
  ``{a, cb, ca}``. Lane neighbors must be on the same refinement level
  (as for registered boundary ops, cross-level exchanges are Parthenon's). Parthenon sorts each list by (``channel_me``, gid,
  ``channel_nb``) and stores it as ``NeighborBlock``\ s
  (``MeshBlock::GetLaneNeighbors(lane)``) with ``lane_channel`` and
  ``lane_nb_channel`` set and zero offsets.
- **Variables.** ``Metadata::SetLane`` sets the ``Metadata::Lane`` flag and
  the lane id on the variable and its flux. A lane variable cannot be
  ``FillGhost`` or ``Sparse``. It is exchanged only in
  ``BoundaryType::lane`` (if it registered boundary ops) and its flux only in
  ``BoundaryType::lane_flxcor_send/recv`` (if it registered a combine op), one
  buffer per (block, lane neighbor, variable). Lane buffers have no geometric
  index set, so the pack ops are required and provide
  ``static std::size_t BufferSize(const MeshBlock &, const NeighborBlock &,
  const Variable<Real> &)``; both ends of a channel must get the same size.
  ``BufferSize`` is the buffer's capacity: a lane pack op may be
  variable-length (see `Variable-length payloads`_) and send fewer elements
  each exchange; to send more than the capacity, return a larger
  ``BufferSize`` and call ``Mesh::RebuildLane`` (``BufferSize`` is called
  only when the lane's buffers are built). Flux-correction pack ops always
  send whole buffers.
  ``BndInfo::channel`` and ``BndInfo::nb_channel`` hold the channel ids on the
  local block and on the neighbor; the buckets are sorted by (``block``,
  variable label, ``channel``, neighbor gid), so the buffers arriving on one
  channel are consecutive and in ascending sender gid.
- **Tasks.** ``AddLaneExchangeTasks`` adds send, receive and ``SetBounds``
  (after the send of the same ``MeshData``); ``AddLaneFluxCorrectionTasks``
  the same for the flux. They exchange only the variables of the given lane;
  the ``MeshData`` may also hold variables of other lanes and ordinary
  variables, so one ``MeshData`` can be exchanged lane by lane (and with the
  geometric exchange). The task functions are ``SendLaneBoundBufs``,
  ``StartReceiveLaneBoundBufs``, ``ReceiveLaneBoundBufs`` and
  ``SetLaneBounds``, templated on the lane ``BoundaryType`` and taking the
  ``LaneId``. Coalesced comms are not supported on lanes.
- **Caches.** The boundary cache of a ``MeshData`` keeps one sub-cache per
  geometric ``BoundaryType`` (as before) and, for lanes, one per
  (``BoundaryType``, send/receive, lane), created when the lane's tasks are
  added; each holds only its lane's boundaries.
- **Rebuild.** ``Mesh::RebuildLane(lane)`` is rank-local: it reruns the
  provider for every local block and rebuilds the lane's neighbor lists, tag
  map and buffers, and bumps the lane's own comm-map epoch: only that lane's
  sub-caches (in every ``MeshData``) re-initialize at their next exchange.
  Other lanes and the geometric exchanges keep their buffers and caches.
  Every rank on either end of a changed
  channel must call it before its next exchange on the lane, and the lane
  must be quiescent (no lane message of a previous exchange in flight, no
  receive posted), i.e. call it between exchanges. Lanes are rebuilt
  automatically with the other buffers when the mesh is (load balancing,
  remeshing), which bumps the global epoch and so re-initializes every cache.
  ``RegisterNeighborProvider`` called after the mesh was built
  builds the lane right away.
- **Cost.** Without a registered provider nothing is built or run for lanes:
  no lane neighbor lists, tag maps, buffers or tasks. The geometric exchanges
  gain no branch in their pack and unpack kernels, and the ``BoundaryType``
  indexed cache array keeps its size (the lane sub-caches live in a map that
  stays empty).

The ``provider_lanes`` example has a pool of 8 x 8 blocks and two lanes whose
graphs ignore the block positions: a groups lane, where in each of three
layers the blocks are shuffled into groups of 2 to 6 (each block is a lane
neighbor of the other members, on channel ``(layer + gid) % 3``, so the two
ends of a channel use different ids), and a ring lane over a shuffled block
order. One ``MeshData`` holds both lanes' variables and an ordinary
``FillGhost`` variable, and is exchanged lane by lane and with the ghost
exchange. Per-block records are exchanged with an unpack op that sums the
buffers of each channel in ascending gid, and a face flux with a combine op
that replaces each block's candidate by the group mean. A third record on the
groups lane uses a variable-length pack op: each sender sends a number of rows
that depends on its gid, channel and the cycle, at most the capacity (a
package parameter returned by ``BufferSize``, raised together with the
rebuild below); the unpack op reads the received lengths on device and stores,
per channel, the sum of the rows and their number. The library ops
``boundary_ops::AverageOverlapping`` and ``flux_correction_ops::AverageShared``
locate elements through the geometric index sets and coordinate transforms of a
``BndInfo``, which lane buffers do not have, so the example keeps its own ops. Before cycle 2 the
groups are reshuffled and the lane rebuilt; the example counts the cache
initializations and checks that only the groups lane's caches were
re-initialized. Every cycle is checked bitwise on host against the graph; the
results are identical on 1 to 4 ranks, and
``provider_lanes/rebuild=false`` shows the check failing.
