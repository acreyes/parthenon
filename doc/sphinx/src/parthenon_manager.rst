.. _parthenonmanager:

Parthenon Manager
=================

The ``ParthenonManager`` class helps set up a parthenon-based
application. An instance of ``ParthenonManager`` owns pointers a
number of sub-objects:

* The ``ApplicationInput`` struct, which lets users set things like
  the ``ProcessPackages`` and ``ProblemGenerator`` function pointers.
* The ``ParameterInput`` class, which populates input parameters from
  the input file and command line
* The ``Mesh`` object

The ``ParthenonManager`` has two important methods that usually must
be called in the ``main`` function of a parthenon-based app. The
function

.. code:: cpp

   ParthenonStatus ParthenonManager::ParthenonInitEnv(int argc, char *argv);

reads the input deck and populates the ``ParameterInput`` object
pointer ``pman.pin``, and sets up the ``MPI``, and ``Kokkos``
runtimes. The function

.. code:: cpp

   void ParthenonManager::ParthenonInitPackagesAndMesh();

Calls the ``Initialize(ParameterInput *pin)`` function of all packages
to be utilized and creates the grid hierarchy, including the ``Mesh``
and ``MeshBlock`` objects, and calls the ``ProblemGenerator`` (and
``PostProblemGenerator`` and ``PostInitialization``) routines.

The reason these functions are split out is to enable decisions to be
made by the application between reading the input deck and setting up
the grid. For example, during problem initialization, ``ProblemGenerator``
may be used to be the user-facing API to describe initial conditions,
whereas, ``PostProblemGenerator`` could use those user-specified fields
to sync *all* fields prior to entering communication routines.
``PostInitialization`` could be used to make any last initialization changes
after the initialization hierarchy has converged, regardless of whether that
required AMR, SMR, or no refinement at all. A common use-case is:

.. code:: cpp

  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = MyProcessPackages;
  std::string prob = pman.pin->GetString("app", "problem");
  if (prob == "problem1") {
    pman.app_input->ProblemGenerator = Problem1Generator;
    pman.app_input->PostProblemGenerator = Problem1PostProblemGenerator;
  } else {
    pman.app_input->ProblemGenerator = Problem2Generator;
  }

  pman.ParthenonInitPackagesAndMesh();

Initializing from a ``ParameterInput``
--------------------------------------

Instead of parsing the command line and reading an input file, an
application can build the ``ParameterInput`` itself (e.g., from another
input format or from Python) and pass it to

.. code:: cpp

   ParthenonStatus ParthenonManager::ParthenonInitEnv(std::unique_ptr<ParameterInput> pin,
                                                      RuntimeOptions opts);

``RuntimeOptions`` (``runtime_options.hpp``) holds the settings that otherwise come
from the command line flags, other than the input file:

=====================  ===========  ==========================================
Member                 Flag         Meaning
=====================  ===========  ==========================================
``restart_filename``   ``-r``       Restart from this file
``analysis``           ``-a``       Only analyze ``restart_filename``
``run_dir``            ``-d``       Run directory
``params_regex``       ``-p``       Report parameters of matching blocks and quit
``mesh_flag``          ``-m``       Output the mesh structure and quit
``wall_time_limit``    ``-t``       Wall time limit in seconds
``watchdog_timeout``   ``-w``       Watchdog timeout in seconds
``kokkos_args``                     Arguments passed to ``Kokkos::initialize``
=====================  ===========  ==========================================

MPI and Kokkos are only initialized if they are not already initialized (and
``ParthenonFinalize`` only finalizes Kokkos if Parthenon initialized it). On a restart,
the parameters stored in the restart file are loaded first and ``pin`` is overlaid on
top of them (``ParameterInput::Overlay``), just like an input file passed together with
``-r``. ``ParthenonInitEnv(argc, argv)`` is implemented on top of this overload: it
converts the command line into ``RuntimeOptions`` and a ``ParameterInput`` read from the
input and/or restart file, with command line overrides applied.

.. code:: cpp

  auto pin = std::make_unique<parthenon::ParameterInput>();
  std::istringstream input("<parthenon/mesh>\nnx1 = 64\n...");
  pin->LoadFromStream(input);

  parthenon::RuntimeOptions opts;
  opts.run_dir = "output";
  auto manager_status = pman.ParthenonInitEnv(std::move(pin), opts);

See :ref:`python` for using this from Python.

.. note::
    This file was made in part with generative AI.
