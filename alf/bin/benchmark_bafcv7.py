# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
"""Reproducible synthetic V7 benchmarks and four-GPU correctness smoke test.

Example:
  OMP_NUM_THREADS=1 python -m alf.bin.benchmark_bafcv7 --output /tmp/v7.json
  OMP_NUM_THREADS=1 python -m alf.bin.benchmark_bafcv7 --ddp-smoke

Timing includes forward, loss, backward, Adam, and target updates, but excludes
MuJoCo/replay sampling. JSON reports iteration latency, memory and seed reuse.
"""
import argparse
import copy
import itertools
from datetime import timedelta
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import alf
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.utils.bafcv7_benchmark_utils import (
    FLAGS, make_algorithm, make_batch, compute_loss)

CASES = {
    'reference': (),
    'probe_cache': FLAGS[:1],
    'seed_dedup': FLAGS[1:2],
    'selective_critic': FLAGS[2:3],
    'shared_observation': FLAGS[3:4],
    'combined': FLAGS
}


def benchmark(args):
    alf.set_default_device('cuda')
    torch.cuda.set_device(args.device)
    results = []
    for variant in args.variants:
        for mode in args.features:
            for unique in args.unique:
                for case in args.cases:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.manual_seed(20260926)
                    alg = make_algorithm(
                        case != 'reference', variant, mode, source=args.probes,
                        large=True, flags=CASES[case])
                    # Construction preserves initialization RNG/parameter order.
                    alg._critic_update_counter = 1
                    alg._apply_train_mode_grad_flags()
                    optimizer = torch.optim.Adam(alg.parameters(), lr=3e-4)
                    batches = [make_batch(alg, 128, unique) for _ in range(12)]
                    durations = []

                    def iteration(index):
                        for step in range(12 if args.cycle else 1):
                            if not args.cycle:
                                alg._train_mode = TrainMode.critic
                                alg._apply_train_mode_grad_flags()
                            optimizer.zero_grad(set_to_none=True)
                            loss, info = compute_loss(
                                alg, batches[(index + step) % 12])
                            loss.backward()
                            optimizer.step()
                            alg.after_update(None, info)

                    for i in range(args.warmup):
                        iteration(i)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start_hits = alg._probe_cache_hits
                    for i in range(args.iterations):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        iteration(i)
                        torch.cuda.synchronize()
                        durations.append(time.perf_counter() - start)
                    row = dict(
                        variant=variant, features=mode, case=case,
                        unique_seeds=unique, replay_items=128, probes=args.probes,
                        updates_per_iteration=12 if args.cycle else 1,
                        median_seconds=statistics.median(durations),
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        probe_cache_hits=alg._probe_cache_hits - start_hits,
                        warmup=args.warmup, iterations=args.iterations,
                        seed_ratio=unique / 128)
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    Path(args.output).write_text(
                        json.dumps(results, indent=2) + '\n')
                    del optimizer, alg, batches
    return results


def check_gpu_equivalence(device):
    """Compare dog-sized float32 losses and every parameter gradient."""
    alf.set_default_device('cuda')
    torch.cuda.set_device(device)
    for variant, mode, phase in itertools.product(
            ['ensemble_base', 'single_seeded'],
            ['mean_log_std', 'action_quantiles'], ['initial', 'critic', 'actor']):
        torch.manual_seed(23)
        ref = make_algorithm(False, variant, mode, large=True)
        fast = make_algorithm(True, variant, mode, large=True)
        fast.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
        for alg in (ref, fast):
            if phase != 'initial':
                alg._critic_update_counter = 1
                alg._train_mode = (TrainMode.critic if phase == 'critic'
                                   else TrainMode.actor)
                alg._apply_train_mode_grad_flags()
        batch = make_batch(ref, 128, 32)
        rng = torch.cuda.get_rng_state()
        ref_loss, _ = compute_loss(ref, batch)
        ref_loss.backward()
        torch.cuda.set_rng_state(rng)
        fast_loss, _ = compute_loss(fast, batch)
        fast_loss.backward()
        torch.testing.assert_close(ref_loss, fast_loss, rtol=3e-4, atol=3e-6)
        maximum_error = 0.
        for (name, a), (other, b) in zip(
                ref.named_parameters(), fast.named_parameters()):
            assert name == other
            if a.grad is None or b.grad is None:
                assert a.grad is b.grad, name
            else:
                torch.testing.assert_close(
                    a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)
                maximum_error = max(
                    maximum_error, (a.grad - b.grad).abs().max().item())
        print(dict(variant=variant, features=mode, phase=phase,
                   maximum_gradient_error=maximum_error), flush=True)
        del ref, fast, alg, a, b, ref_loss, fast_loss, batch
        gc.collect()
        torch.cuda.empty_cache()


def _ddp_loss(alg, batch):
    loss, _ = compute_loss(alg, batch)
    return loss


def _ddp_worker(rank, world, rendezvous, directory):
    from alf.utils.distributed import make_ddp_performer
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    alf.set_default_device('cuda')
    # Match alf.bin.train's single-host multi-GPU backend.
    dist.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=world,
        timeout=timedelta(seconds=60))
    try:
        for variant in ('ensemble_base', 'single_seeded'):
            for mode in ('mean_log_std', 'action_quantiles'):
                torch.manual_seed(22)
                alg = make_algorithm(True, variant, mode)
                # Register all parameters before the first train-mode switch.
                runner = make_ddp_performer(
                    alg, _ddp_loss, find_unused_parameters=True)
                optimizer = torch.optim.Adam(alg.parameters(), lr=3e-4)
                names = tuple(n for n, _ in alg.named_parameters())
                for step in range(12):
                    torch.manual_seed(1000 + rank * 100 + step)
                    batch = make_batch(alg, unique=1 + rank)
                    optimizer.zero_grad(set_to_none=True)
                    critic_only = alg._critic_only()
                    cached = alg._probe_cache
                    hits, misses = alg._probe_cache_hits, alg._probe_cache_misses
                    loss = runner(batch)
                    if critic_only:
                        if cached is None:
                            assert alg._probe_cache_misses == misses + 1
                        else:
                            assert alg._probe_cache is cached
                            assert alg._probe_cache_hits == hits + 1
                            assert alg._probe_cache_misses == misses
                    loss.backward()
                    assert torch.isfinite(loss)
                    # Include unused slots when checking synchronized gradients.
                    gradients = torch.cat([
                        (torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                        for p in alg.parameters()
                    ])
                    expected = gradients.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(
                        gradients, expected, rtol=1e-5, atol=1e-6)
                    optimizer.step()
                    alg.after_update(None, None)
                    assert names == tuple(n for n, _ in alg.named_parameters())
                    params = torch.cat([
                        p.detach().flatten() for p in alg.parameters()
                    ])
                    expected = params.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(
                        params, expected, rtol=1e-5, atol=1e-6)
                    if step == 5:
                        path = Path(directory) / f'{variant}_{mode}_{rank}.pt'
                        torch.save(dict(model=alg.state_dict(),
                                        optimizer=optimizer.state_dict()), path)
                        checkpoint = torch.load(path, weights_only=False)
                        alg.load_state_dict(checkpoint['model'], strict=True)
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        assert alg._probe_cache is None
                dist.barrier()
                if rank == 0:
                    print(f'DDP passed: {variant} {mode}; '
                          f'probe cache hits={alg._probe_cache_hits}, '
                          f'misses={alg._probe_cache_misses}', flush=True)
                del runner, optimizer, alg
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    variants = ['ensemble_base', 'single_seeded']
    features = ['mean_log_std', 'action_quantiles']
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--variants', nargs='+', default=variants, choices=variants)
    parser.add_argument('--features', nargs='+', default=features, choices=features)
    parser.add_argument('--unique', type=int, nargs='+', default=[1, 8, 32, 64, 128])
    parser.add_argument('--cases', nargs='+', default=list(CASES), choices=list(CASES))
    parser.add_argument('--probes', choices=['frozen', 'trainable'], default='frozen')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--cycle', action='store_true',
                        help='Time complete 12-update cycles')
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--ddp-smoke', action='store_true')
    validation.add_argument('--check-equivalence', action='store_true')
    parser.add_argument('--output', default='/tmp/bafcv7_benchmark.json')
    args = parser.parse_args()
    if any(u < 1 or u > 128 for u in args.unique):
        parser.error('unique seeds must be 1..128')
    if args.warmup < 0 or args.iterations < 1:
        parser.error('invalid iteration count')
    if args.check_equivalence:
        check_gpu_equivalence(args.device)
    elif args.ddp_smoke:
        with tempfile.TemporaryDirectory(prefix='bafcv7_ddp_') as directory:
            mp.spawn(
                _ddp_worker,
                args=(4, 'file://' + directory + '/rendezvous', directory),
                nprocs=4, join=True)
    else:
        benchmark(args)


if __name__ == '__main__':
    main()
