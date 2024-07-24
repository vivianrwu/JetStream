# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AOT compilation utils."""

import jax
import jax.numpy as jnp
from jax.experimental import layout as jax_layout
import concurrent.futures
from typing import Any, Optional, cast
import logging
import frozendict
from jetstream.engine import engine_api, token_utils

XLAFlags = frozendict.frozendict({
    "xla_tpu_enable_data_parallel_all_reduce_opt": True,
    "xla_tpu_data_parallel_opt_different_sized_ops": "true",
    "xla_tpu_enable_async_collective_fusion": True,
    "xla_tpu_enable_async_collective_fusion_fuse_all_gather": "true",
    "xla_tpu_enable_async_collective_fusion_multiple_steps": True,
    "xla_tpu_overlap_compute_collective_tc": True,
    "xla_enable_async_all_gather": "true",
})

DLL = jax_layout.DeviceLocalLayout
Layout = jax_layout.Layout

def make_shaped_array(
    t: Any, sharding: None | Any = None
):
    if hasattr(t, 'sharding'):
        return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=t.sharding)
    else:
        return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=sharding)

def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
) -> bool:
  """Organizes the engines and executables.

  Args:
      prefill_engines: Prefill only engines.
      generate_engines: Generate only engines.
      prefill_params: Prefill only params.
      generate_params: Generate only params.
  """

  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []

  any_prefill_engine = None
  any_prefill_params = None

  prefill_executables = []
  inserts_generate_executables = []

  for i, pe in enumerate(prefill_engines):
    any_prefill_engine = pe
    any_prefill_params = prefill_params[i]
    prefill_executable = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
    )
    prefill_executables.append(prefill_executable)

  for i, ge in enumerate(generate_engines):
    insert_executable, generate_executable = (
        initialize_insert_generate_jit_cache(
            prefill_engine=any_prefill_engine,
            generate_engine=ge,
            prefill_params=any_prefill_params,
            generate_params=generate_params[i],
            generate_idx=i,
        )
    )
    inserts_generate_executables.append(
        [insert_executable, generate_executable]
    )

  if prefill_executables and inserts_generate_executables:
#   if prefill_executables:
    return True
  return False


def get_optimal_prefill_layouts(
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
):
    layouts = DLL.AUTO
    prefill_params = jax.tree.map(make_shaped_array, prefill_params)

    prefill_out_layouts = (
        jax.tree.map(
            lambda s: Layout(layouts, s),
            prefill_engine.get_prefix_destination_sharding(),
        ),
        Layout(layouts, prefill_engine.replicated_sharding),
    )

    padded_tokens, true_length = jnp.ones((prefill_engine.max_prefill_length), dtype="int32"), prefill_engine.max_prefill_length

    prefill_with_layout = jax.jit(
        prefill_engine._downstream_engine.prefill,
        in_shardings=Layout(layouts),
        out_shardings=prefill_out_layouts,
    )
    lowered_prefill = prefill_with_layout.lower(
        params=prefill_params, 
        padded_tokens=padded_tokens, 
        true_length=true_length
    )
    compiled_prefill = lowered_prefill.compile(
        compiler_options=XLAFlags
    )
    arg_layouts, _ = compiled_prefill.input_layouts()
    logging.info("arg_layouts is %s", arg_layouts)
    return arg_layouts

def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    prefill_idx: int,
):
  """Precompile all prefill functions in parallel.
  If we don't do this, then when a new request triggers a new prefill bucket it
  will take a very long time for that query to come back.

  Args:
      prefill_engine: A prefill engine to be compiled for.
      prefill_params: The associated prefill parameters.
      prefill_idx: Which prefill engine it is.
  """
  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= prefill_engine.max_prefill_length
  ]
  prefill_engine.prefill_buckets = prefill_buckets
  if prefill_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(prefill_engine.max_prefill_length)

#   prefill_param_layouts = get_optimal_prefill_layouts(
#     prefill_engine, prefill_params
#   )

  layouts = DLL.AUTO
  prefill_out_layouts = (
    jax.tree.map(
        lambda s: Layout(layouts, s),
        prefill_engine.get_prefix_destination_sharding(),
    ),
    Layout(layouts, prefill_engine.replicated_sharding),
)

  param_shapes = jax.tree.map(make_shaped_array, prefill_params)

  def compile_prefill(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    # prefill = jax.jit(
    #     prefill_engine._downstream_engine.prefill,  # pylint: disable=protected-access
    #     in_shardings=(prefill_param_layouts, None, None, None),
    #     out_shardings=(
    #         prefill_engine.get_prefix_destination_sharding(),
    #         prefill_engine.replicated_sharding,
    #     ),
    # )
    prefill_with_layout, _ = jax.jit(
        prefill_engine._downstream_engine.prefill,
        in_shardings=Layout(layouts),
        out_shardings=prefill_out_layouts,
    )
    lowered = prefill_with_layout.lower(
        params=param_shapes,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    logging.info(
        "---------Prefill engine %d lowered for prefill length %d.---------",
        prefill_idx,
        length,
    )
    compiled = lowered.compile(compiler_options=XLAFlags)
    logging.info(
        "---------Prefill engine %d compiled for prefill length %d.---------",
        prefill_idx,
        length,
    )
    return compiled

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    prefill_executable = list(executor.map(compile_prefill, prefill_buckets))

  prefill_executable = {
      k: cast(jax.stages.Compiled, e)
      for k, e in zip(prefill_buckets, prefill_executable)
  }

  prefill_engine.prefill_executable = prefill_executable
  prefill_engine.warm = True

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_executable


def initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    generate_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    generate_params: Any,
    generate_idx: int,
):
  """Initialiszes jit cache for insert and generate.

  Args:
      generate_engine: A generate engine to be compiled for.
      generate_params: The associated parameters.
      generate_idx: Which generate engine it is.
  """

#   layouts = Layout(DLL.AUTO)
  decode_state = generate_engine.init_decode_state()
  decode_state_shapes = jax.tree.map(make_shaped_array, decode_state)

  prefill_param_shapes = jax.tree.map(make_shaped_array, prefill_params)
  generate_param_shapes = jax.tree.map(make_shaped_array, generate_params)

  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= generate_engine.max_prefill_length
  ]
  generate_engine.prefill_buckets = prefill_buckets
  if generate_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(generate_engine.max_prefill_length)

#   decode_state = generate_engine.decode_state

  def compile_insert(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    prefill, _ = prefill_engine._downstream_engine.prefill(  # pylint: disable=protected-access
        params=prefill_param_shapes,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    slot_shape = jax.ShapeDtypeStruct(
        (),
        jnp.int32,
        sharding=generate_engine.replicated_sharding,
    )

    insert_with_layout = jax.jit(
        generate_engine._downstream_engine.insert,
        in_shardings=(None, decode_state_layouts, None, None),
        out_shardings=decode_state_layouts,
        donate_argnums=(1,),
    )
    insert_lowered = insert_with_layout.lower(
        prefill, decode_state_shapes, slot_shape
    )

    # lowered = jax.jit(generate_engine._downstream_engine.insert).lower(  # pylint: disable=protected-access
    #     prefix=prefill, decode_state=decode_state, slot=slot_shape
    # )
    logging.info(
        "---------Generate engine %d lowered for insert length %d.---------",
        generate_idx,
        length,
    )
    # compiled = lowered.compile(compiler_options=XLAFlags)

    insert_executable = insert_lowered.compile(
        compiler_options=XLAFlags
    )
    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )
    return insert_executable

  def compile_generate_and_get_layouts(
    generate_engine: engine_api.JetStreamEngine,
    generate_params: Any,
    decode_state: Optional[Any] = None,
  ):

    param_layout = Layout(DLL.AUTO)
    decode_state_layout = Layout(DLL.AUTO)

    generate_out_layouts = (
        jax.tree.map(
            lambda s: Layout(decode_state_layout.device_local_layout, s),
            generate_engine.get_decode_state_sharding(),
        ),
        Layout(
            decode_state_layout.device_local_layout,
            generate_engine.replicated_sharding,
        ),
    )
    generate_with_layout = jax.jit(
        generate_engine._downstream_engine.generate,
        in_shardings=(param_layout, decode_state_layout),
        out_shardings=generate_out_layouts,
        donate_argnums=(1,),
    )
    lowered_generate = generate_with_layout.lower(generate_params, decode_state)

    compiled_generate = lowered_generate.compile(
        compiler_options=XLAFlags
    )

    arg_layouts, _ = compiled_generate.input_layouts()  # pylint:disable = protected-access
    logging.info("arg_layouts for generate:")
    logging.info(arg_layouts)
    return compiled_generate, arg_layouts[0], arg_layouts[1], generate_out_layouts


    # logging.info(
    #     "---------Generate compilation %d begun.---------", generate_idx
    # )

    # lowered = jax.jit(generate_engine._downstream_engine.generate).lower(  # pylint: disable=protected-access
    #     params=generate_params,
    #     decode_state=decode_state,
    # )
    # logging.info(
    #     "---------Generate engine %d lowered.---------",
    #     generate_idx,
    # )

    # compiled = lowered.compile(compiler_options=XLAFlags)
    # logging.info(
    #     "---------Generate engine %d compiled.---------",
    #     generate_idx,
    # )

    # logging.info(
    #     "---------Generate compilation %d complete.---------", generate_idx
    # )

    # return compiled

  def compile_init_decode_state():
    init_decode_state_with_layout = jax.jit(
        generate_engine.init_decode_state,
        in_shardings=layouts,
        out_shardings=decode_state_layouts,
    )
    lowered = init_decode_state_with_layout.lower()
    executable = lowered.compile(compiler_options=XLAFlags)
    logging.info(
        '---------Generate engine %d compiled init decode state.---------',
        generate_idx,
    )
    return executable

  logging.info(
      "---------Insertion generation compilation %d begun.---------",
      generate_idx,
  )

  (
      generate_executable,
      param_layouts,
      decode_state_layouts,
      generate_out_layouts,
  ) = compile_generate_and_get_layouts(
      generate_engine,
      generate_param_shapes,
      decode_state_shapes,
  )

#   generate_executable = compile_generate()
  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )
  generate_engine.generate_executable = generate_executable

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    insert_executable = list(executor.map(compile_insert, prefill_buckets))
    init_decode_state_executable = executor.submit(compile_init_decode_state)

  insert_executable = {
      k: cast(jax.stages.Compiled, e)
      for k, e in zip(prefill_buckets, insert_executable)
  }
  generate_engine.insert_executable = insert_executable
  generate_engine.init_decode_state_executable = init_decode_state_executable
  generate_engine.warm = True

  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return insert_executable, generate_executable
