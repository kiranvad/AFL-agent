Collect variables from multiple records
======================================

``setup_data_collection`` configures the input dataset; ``append_dataset`` adds
one sample using exact Tiled entry IDs. Each key in ``sources`` names an output
variable. Existing single-path specs remain supported and can be mixed with
constructed variables.

A constructed variable declares named ``inputs`` with driver/task selectors and
``source`` set to ``metadata`` or ``dataset``. Its ``mapping`` assigns each output
field an input name and extraction path. Metadata paths traverse nested
dictionaries using dots; dataset paths select an xarray variable.

Set ``mapping_to_vector=True``, declare one feature dimension in ``dims``, and
list all mapped field names exactly once in ``coords`` for that dimension. This
coordinate list determines vector order regardless of mapping insertion order.
Mapped values must be numeric scalars. The sample dimension is added automatically.

For example, combine square-wave voltammetry settings with a chronoamperometry
duration from the same sample:

.. code-block:: python

    input_spec = {
        "campaign_id": "asv-check-b71d1f7e",
        "sample_dim": "sample",
        "sources": {
            "parameters": {
                "inputs": {
                    "swv": {
                        "driver_name": "GamryProxyDriver",
                        "task_name": "runSWV",
                        "campaign_id_path": "attrs.AL_campaign_name",
                        "sample_id_path": "attrs.sample_uuid",
                        "source": "metadata",
                    },
                    "ca": {
                        "driver_name": "GamryProxyDriver",
                        "task_name": "runCA",
                        "campaign_id_path": "attrs.AL_campaign_name",
                        "sample_id_path": "attrs.sample_uuid",
                        "source": "metadata",
                    },
                },
                "mapping": {
                    "pulse_size": {
                        "input": "swv", "path": "attrs.parameters.pulse_size",
                    },
                    "step_size": {
                        "input": "swv", "path": "attrs.parameters.step_size",
                    },
                    "frequency": {
                        "input": "swv", "path": "attrs.parameters.frequency",
                    },
                    "step2_time": {
                        "input": "ca", "path": "attrs.parameters.step2_time",
                    },
                },
                "mapping_to_vector": True,
                "dims": ["parameter"],
                "coords": {
                    "parameter": ["pulse_size", "step_size", "frequency", "step2_time"],
                },
            },
        },
    }
    agent.enqueue(
        task_name="setup_data_collection",
        input_spec=input_spec,
        interactive=True,
    )
    agent.enqueue(
        task_name="append_dataset",
        entries={
            "parameters": {
                "swv": "QD-f9be96d9-a7b5-4886-9984-3bce9a5d422a",
                "ca": "QD-3fb9b181-a64a-4177-82ac-81cba0c7f382",
            },
        },
        interactive=True,
    )

This creates ``parameters(sample, parameter)`` with values
``[[0.025, 0.002, 25.0, 60.0]]``. These paths select the recorded measurement
parameters, which may differ from the defaults under ``attrs.driver_config``.
Each input can use a different driver and data source type.

``entries`` must contain every configured source. A constructed source accepts
a dictionary containing exactly its input names; a single-path source accepts a
plain entry ID string. All records must match their selectors, belong to the
configured campaign, and have the same sample ID. Duplicate samples are rejected.
Validation and extraction errors leave the accumulated input dataset unchanged.

Provenance coordinates are named ``<variable>_<input>_entry_id`` for constructed
sources, and ``<variable>_entry_id`` for single-path sources. Each coordinate is
indexed by sample. Avoid names that produce identical provenance coordinates.
Repeated references to the same entry within an append share the entry lookup
and dataset read.
