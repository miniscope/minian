Working with MiniAn Variables
=============================

As :ref:`noted in the pipeline <pipeline/notebook_2:loading videos and visualization>`, almost all MiniAn array variables are :py:class:`xarray.DataArray` objects, which are just n-dimensional arrays with extra metadata.
We highly recommend taking a look at the :doc:`xarray:user-guide/data-structures` documentation to get familiar with this format.
We also recommend taking a look at the :ref:`pipeline/notebook_2:subset part of video` part in the pipeline, as well as the :doc:`xarray:user-guide/indexing` documentation to understand how to index and manipulate the data.
Lastly, the actual data (without coordinates/metadata) in MiniAn are almost always represented as :doc:`dask array <dask:array>`, which are essentially container of a bunch of tasks that can be executed to obtain the data, instead of actual data in RAM.
To obtain actual in-memory representation of data, you can always call the :py:meth:`.compute() <xarray:xarray.DataArray.compute>` function on any variable, which would convert the underlying data representation into a plain :doc:`numpy ndarray <numpy:reference/arrays.ndarray>`.
Note that this implies loading the data into RAM so you should make sure you have enough available RAM for such operation.

Here are some additional tips on working with this format:

Custom checkpoints
------------------

If for whatever reason you have to restart the python kernel in a run of pipeline, then all the variables in RAM would be lost.
Luckily chances are you don't have to restart from the beginning.
Whenever you see a successful call to :py:func:`save_minian <minian.utilities.save_minian>`, that variable is stored to disk (either at ``dpath`` or ``intpath``). And you can just load it to pick up where you're left off.

If you feel the need, you can always add more :py:func:`save_minian <minian.utilities.save_minian>` call.
One thing that's potentially puzzling is that the name of each arrays stored in the dataset is not necessarily the same as your variable name.
Instead it's the :py:meth:`.name <xarray:xarray.DataArray.name>` attribute of each array, which can be modified with :py:meth:`.rename() <xarray:xarray.DataArray.rename>` function.
This is why in the default pipeline you always see :py:meth:`.rename() <xarray:xarray.DataArray.rename>` call on each array saved with :py:func:`save_minian <minian.utilities.save_minian>` to make sure things are not too confusing.
However you can always change the name of the saved array.
For example, to save different versions of the ``C`` array in different iterations of CNMF (so that they don't overwrite each other), you can:

.. code-block:: python

    save_minian(C.rename("C_matrix_first_iteration"), ...)

Then afterwards you can load it with the name ``"C_matrix_first_iteration"``.

At any point, you can use the :py:func:`open_minian <minian.utilities.open_minian>` function to load existing dataset on disk.
For example, to load data from the intermediate path, and assign the names ``Y_fm_chk``, ``A``, as well as using the data we saved from the code block above as ``C``, you can:

.. code-block:: python

    minian_ds = open_minian(intpath)
    Y_fm_chk = minian_ds["Y_fm_chk"]
    A = minian_ds["A"]
    C = minian_ds["C_matrix_first_iteration"]

There are some additional "gotcha":

* By default :py:func:`open_minian <minian.utilities.open_minian>` will try to align the saved data based on dimension/coordinates.
  If for some reason the metadata are not compatible, for example if you have some variables saved from a different sessions under the same path, this could result in unnecessary NaN-padding or even exceptions.
  Ideally you should clean up the data folder between different runs of pipeline and make sure metadata are handled properly.
  However if you just want to load the data without dealing with the mess, you can always use the ``return_dict`` argument like:

  .. code-block:: python

        minian_ds = open_minian(..., return_dict=True)

* The chunk size across variables should be consistent within a run of the pipeline (see :ref:`tips/performance:Chunked computation` for more detail).
  This is controled by the ``chk`` dictionary, which is assigned :ref:`at the beginning of the pipeline <pipeline/notebook_2:loading videos and visualization>`.
  If you need ``chk`` after restarting the python kernel, you can run that cell again.
  Alternatively you can manually note down the content of ``chk`` and potentially add a line like the following to execute everytime:

  .. code-block:: python

        chk = {"frame": ..., "height": ..., "width": ...}

* Since both spatial and temporal update of CNMF may drop cells, it's important to keep the cell labels consitent across the spatial component ``A`` and temporal component ``C`` and ``S``.
  However, it's very inefficient to re-save the component that's not actually updated every time.
  Instead we can just use the cell labels on the up-to-date variables to subset data.
  Hence you might see something like this during the saving steps of CNMF:

  .. code-block:: python

        A = save_minian(A_new.rename("A"), ...)
        C = C.sel(unit_id=A.coords["unit_id"].values)

  Because the subsetted version of ``C`` is not saved, if later you want to continue from this stage, be sure to do the subsetting again:

  .. code-block:: python

        minian_ds = open_minian(intpath)
        A = minian_ds["A"]
        C = minian_ds["C"]
        C = C.sel(unit_id=A.coords["unit_id"].values)

Saving to other formats
-----------------------

By default MiniAn stores all variables in the :ref:`xarray-augmented zarr <xarray:io.zarr>` format for better support of parallel and out-of-core computation.
Once you are done with the pipeline, however, you might want to convert the outputs to other formats for better compatability with down-stream analysis.
At any point, you can do:

.. code-block:: python

    minian_ds = open_minian(dpath)

and the returned ``minian_ds`` would be a :py:class:`xarray:xarray.Dataset` object.
You would be able to save the whole dataset to any metadata-rich format using the :doc:`xarray io api <xarray:user-guide/io>`.
For example, you can do the following to save the dataset into netCDF format (which, by the way, is supported by :py:func:`open_minian <minian.utilities.open_minian>`):

.. code-block:: python

    minian_ds.to_netcdf("minian_dataset.nc")

If, however, you are more comfortable working with tabular data formats, you can do something like the following to save any individual array to a :ref:`pandas dataframe <pandas:/user_guide/dsintro.rst#dataframe>` in the `"long" data format <https://en.wikipedia.org/wiki/Wide_and_narrow_data>`_:

.. code-block:: python

    minian_ds["C"].rename("C").to_series().reset_index()

where each coordinate in the array would become a individual column, and what you put in ``rename()`` would be the name of the column that contains the values of the array.
Then, all the :doc:`pandas io api <pandas:user_guide/io>` would be available to you.
For example, the following would save the `C` array to csv format:

.. code-block:: python

    minian_ds["C"].rename("C").to_series().reset_index().to_csv("C.csv")

Lastly, if you don't care about metadata and would prefer to work with raw numbers, you can always call the :py:meth:`.values <xarray:xarray.DataArray.values>` attribute on any array to access the underlying numpy array.
Then, all the :doc:`numpy io api <numpy:reference/routines.io>` would be available to you.
For example, the following would save the `C` array to npy format:

.. code-block:: python

    minian_ds["C"].values.save("C.npy")

Again, note that most likely all of these methods involve loading the data into RAM, so you should make sure you have enough free RAM.