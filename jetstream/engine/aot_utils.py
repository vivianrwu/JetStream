"""AOT compilation utils."""

import jax
import jax.numpy as jnp
import concurrent.futures
from typing import Any, Optional
import logging
from jetstream.engine import engine_api, token_utils


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.WarmedUpEngine]] = None,
    generate_engines: Optional[list[engine_api.WarmedUpEngine]] = None,
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

  compiled_prefills = []
  compiled_inserts_generate = []

  for i, pe in enumerate(prefill_engines):
    any_prefill_engine = pe
    any_prefill_params = prefill_params[i]
    prefill_compiled = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
    )
    compiled_prefills.append(prefill_compiled)

  for i, ge in enumerate(generate_engines):
    insert_compiled, generate_compiled = initialize_insert_generate_jit_cache(
        prefill_engine=any_prefill_engine,
        generate_engine=ge,
        prefill_params=any_prefill_params,
        generate_params=generate_params[i],
        generate_idx=i,
    )
    compiled_inserts_generate.append([insert_compiled, generate_compiled])

  if compiled_prefills and compiled_inserts_generate:
    return True
  return False


def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.WarmedUpEngine,
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

  def compile_prefill(length):
    batch_size = prefill_engine.max_concurrent_decodes
    padded_tokens, true_length = jnp.ones((length), dtype='int32'), length

    lowered = jax.jit(
        prefill_engine._downstream_engine.prefill,
        out_shardings=prefill_engine.get_prefix_destination_sharding(),
    ).lower(
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    logging.info(
        "---------Prefill engine %d lowered for prefill length %d.---------",
        prefill_idx,
        length,
    )
    compiled = lowered.compile()
    logging.info(
        "---------Prefill engine %d compiled for prefill length %d.---------",
        prefill_idx,
        length,
    )
    # prefill_compiled[length] = compiled
    return compiled

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

#   prefill_compiled = {}
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    prefill_compiled = list(executor.map(compile_prefill, prefill_buckets))

  prefill_engine.prefill_compiled = prefill_compiled

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_compiled


def initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.WarmedUpEngine,
    generate_engine: engine_api.WarmedUpEngine,
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

  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= generate_engine.max_prefill_length
  ]
  generate_engine.prefill_buckets = prefill_buckets
  if generate_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(generate_engine.max_prefill_length)

  decode_state = generate_engine.init_decode_state()

  def compile_insert(length):
    batch_size = generate_engine.max_concurrent_decodes
    padded_tokens, true_length = jnp.ones((length), dtype='int32'), length

    prefill, first_token = prefill_engine._downstream_engine.prefill(
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    # generate_engine._mesh = generate_engine.mesh
    # slot_shape = jax.ShapeDtypeStruct(
    #     (),
    #     jnp.int32,
    #     sharding=generate_engine.replicated_sharding,
    # )

    lowered = jax.jit(generate_engine._downstream_engine.insert).lower(
        prefix=prefill, decode_state=decode_state, slot=1
    )
    logging.info(
        "---------Generate engine %d lowered for insert length %d.---------",
        generate_idx,
        length,
    )
    compiled = lowered.compile()
    # insert_compiled[length] = compiled

    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )

    return compiled

  def compile_generate():

    logging.info(
        "---------Generate compilation %d begun.---------", generate_idx
    )

    lowered = jax.jit(generate_engine._downstream_engine.generate).lower(
        params=generate_params,
        decode_state=decode_state,
    )
    logging.info(
        "---------Generate engine %d lowered.---------",
        generate_idx,
    )

    compiled = lowered.compile()
    logging.info(
        "---------Generate engine %d compiled.---------",
        generate_idx,
    )

    logging.info(
        "---------Generate compilation %d complete.---------", generate_idx
    )

    return compiled

  logging.info(
      "---------Insertion generation compilation %d begun.---------",
      generate_idx,
  )

  generate_compiled = compile_generate()
  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )
  generate_engine.generate_compiled = generate_compiled

#   insert_compiled = {}
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    insert_compiled = list(executor.map(compile_insert, prefill_buckets))

  generate_engine.insert_compiled = insert_compiled

  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return insert_compiled, generate_compiled
