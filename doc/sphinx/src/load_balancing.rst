.. This file was made in part with generative AI.

.. _load_balancing:

Load Balancing
==============

Parthenon supports some load balancing capabilities. By default load
balancing is done "round robin" so that each MPI rank has a roughly
equal number of ``MeshBlock`` s. However, this behaviour can be
modified.

On a per ``MeshBlock`` basis, you call the
function

.. code:: cpp

   void MeshBlock::SetCostForLoadBalancing(double cost);

where the cost is any positive real number. For example, you might
call the above function in an application driver like:

.. code:: cpp

   for (int pmb : pmy_mesh->block_list) {
     pmb->SetCostForLoadBalancing(SomeHeuristicFunction(pmb));
   }

Then, if you set the following block in your input file:

::

   <parthenon/loadbalancing>
   balancer = manual

parthenon will try to give each MPI rank a set of meshblocks with
equal total cost. To disable this functionality and recover default
behaviour, set the ``balancer`` option to ``default``.

.. note::

   Parthenon does not currently support timer based load balancing,
   however this is a planned feature.

Load-balance groups
-------------------

Ranks always own a contiguous range of gids. To keep runs of
consecutive gids on one rank (for example the blocks that together
make up one element of a downstream data structure), register

.. code:: cpp

   app_in->LoadBalanceGroups =
       [](Mesh *pmesh, const std::vector<LogicalLocation> &locs) {
         std::vector<int> group(locs.size());
         // group[gid] = label of the block's group
         return group;
       };

The function is called on every rank whenever ranks are assigned (mesh
construction, load balancing, remeshing, ``PermuteBlocks``) with the
locations of the new block list (one per gid). It returns a label per
gid, the same on every rank, or an empty vector for no constraint.
Rank boundaries are then placed only where the label changes, at the
group boundary closest to the cost target. The groups must be small
enough that every rank still receives a block. Without the function
nothing changes.

Block permutation
-----------------

For pools of blocks whose position on the mesh carries no physics
(the gids are only slots), the content of the blocks can be permuted
over the gids, e.g. to re-sort them along a space-filling curve of a
moving geometry so that neighbors share a rank:

.. code:: cpp

   void Mesh::PermuteBlocks(const std::vector<int> &new_gid_of_old,
                            ParameterInput *pin, ApplicationInput *app_in);

moves the content of block ``old`` to gid ``new_gid_of_old[old]``: the
remeshed variables (``Independent``, ``FillGhost`` or
``ForceRemeshComm``) with their sparse allocation state, and the block's
load-balance cost. Locations, coordinates, the forest and the gid order
of the mesh are not changed, so the permutation persists through
restarts and later load balancing. Ranks are recomputed for the permuted
costs (and groups), the data moves with the same send/receive as load
balancing, and the neighbors, boundary buffers and all provider lanes are
rebuilt (each neighbor provider is called again, so the downstream graph
must already be expressed in the new gids) before ghosts and derived
fields are filled and the ``base`` ``MeshData`` is recreated. All other
``MeshData`` objects are dropped, as after load balancing. Every rank must
pass the same permutation (computed from global data, so the result is
independent of the rank count); this is checked. Meshes with swarms are
not supported.

To run it automatically, register a callback and a cadence:

.. code:: cpp

   app_in->MeshBlockPermutation =
       [](Mesh *pmesh, ParameterInput *pin, const SimTime &tm) {
         std::vector<int> new_gid_of_old;  // empty: keep the blocks
         // ... update the user's gid-indexed state to the new layout ...
         return new_gid_of_old;
       };

::

   <parthenon/loadbalancing>
   permute_interval = 100

The evolution driver calls the callback after load balancing at the end
of every cycle whose number is a multiple of ``permute_interval``
(default 0: never) and applies the permutation it returns. Without the
callback or with ``permute_interval = 0`` nothing is called.
``example/permute_blocks`` shows a pool whose groups of three blocks are
shuffled every few cycles, with load-balance groups and a provider lane,
and checks that the data follow their blocks.
