Pipeline walkthrough (notebook prose)
=======================================

These subsections mirror markdown cells in ``notebooks/pipeline.ipynb`` so Tips can cross-link without fragile nbsphinx anchors.

.. _tips-pipeline-set-path:

set path and parameters
-----------------------

Set all of the parameters that control the notebook’s behavior.
Ideally, the following cell is the only part of the code the user will have to change when analyzing different datasets.
Here we briefly introduce only some of the initial parameters that are necessary to start the pipeline, and leave the discussion of specific parameters for later.

- ``dpath`` is the folder that contains the videos to be processed.

- ``interactive`` controls whether interactive plots will be shown for parameters exploration.
  Interactive plotting requires CPU/memory usage, and thus could require some time (in particular, those steps where video is played).
  In principle, the user might want to visualize interactive plots during the initial parameters exploration, once the parameters are set and ready for batch processing, the user will set interactive as False to reduce processing time.

- ``output_size`` controls the relative size of all the plots on a scale of 0-100 percent, though it can be set to values >100 without any problem.

- ``param_save_minian`` specifies the destination folder and format of the saved data.
  ``dpath`` is the folder path where the data will be saved.
  ``meta_dict`` is a ``dictionary`` that is used to construct meta data for the final labeled data structure.
  ``overwrite`` is a boolean value controlling whether the data is overwritten if a file already exists.
  We set it to ``True`` here so you can easily play with the demo multiple times, but **use caution** with this option during actual analysis.
  In addition to erasing prior data that may be important to you, overwritting data may cause compatibility issues with existing data from the same minian dataset folder.
  If you want to re-analyze a video from scratch using different parameters, it is recommended that you delete existing data first.

.. note::
   **folder structure**

   The defult ``meta_dict`` in ``param_save_minian`` assumes output minian datasets are stored in heirarchiically arranged folders, as shown below:

   .. code-block:: text

      mice1
      │
      └───session1
      │   │
      │   └───minian
      │       │   Y.zarr
      │       │   A.zarr
      │       │   ...
      │
      └───session2
          │
          └───minian

   The default value can be read as follows:
   The name of the last folder (``-1``) in ``dpath`` (the folder that directly contains the videos) will be used to designate the value of a metadata dimension named ``"session"``.
   The name of the second-to-last folder (``-2``) in ``dpath`` will be used to designate the value for ``"animal"`` and so on.
   Both the keys (name of metadata dimension) and values (numbers indicating which level of folder name should be used) of ``meta_dict`` can be modified to represent your preferred way of data storage.
   Note that the metadata are determined by the folder structure of saved minian datasets, not by those of input movie data.

.. _tips-pipeline-start-cluster:

start cluster
-------------

In ``notebooks/pipeline.ipynb`` the markdown cell for this step is only the heading ``## start cluster``.
The following code cell starts a ``dask.distributed.LocalCluster`` with ``dashboard_address=":8787"``, registers a scheduler plugin, creates a ``dask.distributed.Client``, and prints the dashboard link — open that URL (or `http://localhost:8787/status`) while computations run.

.. _tips-pipeline-loading-videos:

loading videos and visualization
--------------------------------

Recall the values of ``param_load_videos``:

The first argument of load_videos should be the path that contains the videos(``dpath``).
We then pass the dictionary, ``param_load_videos``, defined earlier, which specifies several relevant arguments.
The argument ``pattern`` is optional and is the `regular expression <https://docs.python.org/3/library/re.html>`_ used to filter files under the specified folder.
The default value ``r"msCam[0-9]+\.avi$"`` means that a file can only be loaded if its filename contains **'msCam'**, followed by at least one number, then **'.avi'** as the end of the filename.
This can be changed to suit the naming convention of your videos.
The resulting "video array" ``varr`` contains three dimensions: ``height``, ``width``, and ``frame``.
If you wish to downsample the video, pass in a dictionary to ``downsample``, with the name of dimensions as keys and the downsampling folds as integer value.
The notebook shows temporal downsampling by a factor of 2 with:

.. code-block:: text

    downsample=dict("frame"=2)

``downsample_strategy`` will assume two values: either ``"subset"``, meaning downsampling are carried out simply by subsetting the data, or ``"mean"``, meaning a mean will be calculated on the window of downsampling (the latter being slower).

In addition to the video array ``varr``, the following cell also try to estimate best chunk size ``chk`` to use for computations.
This variable is needed for later steps since it's important to keep chunk size consistent within the pipeline.
If for some reason you have to restart the kernel at some point, remember to either note down the content of ``chk`` or rerun the following cell.

.. note::
   **changing parameters**

   All minian parameters are ``dict`` and you can freely change them in various ways.
   You can go back to the initial parameter setting cell and change things there.
   Alternatively, you can add a code cell before running the relevant step.
   For example, the following line will tell the function to load from ``"/my/data_path"``:

   .. code-block:: python

      param_load_videos["vapth"] = "/my/data_path"

   While the following line will change the downsample setting (which is specified as a ``dict`` on its own) when loading the video:

   .. code-block:: python

      param_load_videos["downsample"] = {"frame": 2}

We then immediately save the array representation to the intermediate folder to avoid repeatedly loading the video in later steps.

The variable ``varr`` is a `xarray.DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray>`_.
Now is a perfect time to familiarize yourself with this data structure and the `xarray <https://xarray.pydata.org/en/stable/>`_ module in general, since we will be using these data structures throughout the analysis.
Basically, a ``xarray.DataArray`` is N-dimensional array labeled with additional metadata, with many useful properties that make them easy to manipulate.
We can ask the computer to print out some information of ``varr`` by calling it (as with any other variable). In the notebook the next cell evaluates ``varr``.

We can see now that ``varr`` is a ``xarray.DataArray`` with a `name <https://xarray.pydata.org/en/stable/generated/xarray.DataArray.name.html#xarray.DataArray.name>`_ ``"demo_movies"`` and three dimensions: ``frame``, ``height`` and ``width``.
Each dimension is labeled with ascending natural numbers.
The `dtype <https://xarray.pydata.org/en/stable/generated/xarray.DataArray.dtype.html#xarray.DataArray.dtype>`_ (`data type <https://docs.scipy.org/doc/numpy-1.14.0/user/basics.types.html>`_) of ``varr`` is ``numpy.uint8``.

Once the data is loaded we can visualize the content of ``varr`` with the help of ``VArrayViewer``, which shows the array as a movie.
You can also plot summary traces like mean fluorescence across ``frames`` by passing a ``list`` of names of traces as inputs.
Currently ``"mean"``, ``"min"``, ``"max"`` and ``"diff"`` are supported, where ``"diff"`` is mean fluorescent value difference across all pixels in a frame.

``VArrayViewer`` also supports a box drawing tool where you can draw a box in the field of view (FOV) and save it as a mask using the ``“save mask”`` button.
The mask is saved as ``vaviewer.mask``, and can be retrieved and used at later stages, for example, when you want to run motion correction on a sub-region of the FOV.
See the `API reference <https://minian.readthedocs.io/page/api/minian.visualization.html#minian-visualization-VArrayViewer>`_ for more detail.

.. _tips-pipeline-subset-video:

subset part of video
--------------------

Before proceeding to pre-processing, it’s good practice to check if there is anything obviously wrong with the video (e.g. the camera suddenly dropped, resulting in dark frames).
This can usually be observed by visualizing the video and plotting the timecourse of the mean fluorescence.
We can utilize the `xarray.DataArray.sel <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.sel.html>`_ method and `slice <https://docs.python.org/3/library/functions.html#slice>`_ to subset any part of the data we want.
By default ``subset = None`` will result in no subsetting.

.. note::
   **subsetting data**

   The `xarray.DataArray.sel <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.sel.html>`_ method takes in either a ``dict`` or keyword arguments.
   In both cases you want to specify the dimension names and the coordinates of the subset as key-value pairs.
   For example, say you want only the first 800 frames of the video, the following two lines will both work and they are equivalent:

   .. code-block:: python

      varr.sel(frame=slice(0, 799))  # slice object is inclusive on both ends
      varr.sel({"frame": slice(0, 799)})

   This also works on arbitrary dimensions.
   For example, the following will give you a 100px x 100px chunk of your movie at a corner:

   .. code-block:: python

      varr.sel(height=slice(0, 99), width=slice(0, 99))
