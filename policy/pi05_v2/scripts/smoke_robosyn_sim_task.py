"""Run one real optimizer step for one Sim task, without saving model checkpoints."""
import argparse
import dataclasses
import functools
import json
from pathlib import Path
import time

import jax
import numpy as np

import train
from openpi.training import config as configs
from openpi.training import data_loader, sharding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    started = time.time()
    config = dataclasses.replace(
        configs.get_config(f'pi05_robosyn_sim_{args.task}_finetune'),
        batch_size=64, num_train_steps=1, fsdp_devices=2, seed=42,
        wandb_enabled=False,
    )
    train.init_logging()
    assert jax.device_count() == 2, jax.devices()
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True, num_batches=1)
    batch = next(iter(loader))
    leaves = jax.tree.leaves(batch)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves), 'Non-finite batch'
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,),
    )
    with sharding.set_mesh(mesh):
        state, info = step(train_rng, state, batch)
    jax.block_until_ready(state)
    metrics = {k: float(v) for k, v in jax.device_get(info).items()}
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert int(state.step) == 1
    result = dict(task=args.task, config=config.name, status='passed', step=0,
                  optimizer_step=int(state.step), batch_size=config.batch_size,
                  fsdp_devices=config.fsdp_devices, seed=config.seed,
                  weights=config.weight_loader.params_path,
                  norm=str(Path(config.data.assets.assets_dir) / config.data.assets.asset_id / 'norm_stats.json'),
                  elapsed_seconds=time.time()-started, **metrics)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.task}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SMOKE_RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
