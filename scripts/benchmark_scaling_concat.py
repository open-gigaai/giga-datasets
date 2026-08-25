import argparse
import bisect
import json
import os
import resource
import runpy
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

EXTRA_PYTHONPATHS = [
    '/gpfs/users/wangyunmo/codes/giga-models/projects/vla/giga-brain-0',
    '/gpfs/users/wangyunmo/codes/giga-datasets-v3.0',
    '/gpfs/users/wangyunmo/codes/giga-models',
    '/gpfs/users/wangyunmo/codes/giga-train',
]


@dataclass
class Stats:
    leaf_open_calls: int = 0
    leaf_first_open_calls: int = 0
    leaf_getitem_calls: int = 0


def ensure_pythonpaths() -> None:
    for path in reversed(EXTRA_PYTHONPATHS):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def load_experiment_config(config_path: str) -> dict[str, Any]:
    scope = runpy.run_path(config_path)
    config = scope['config']
    return dict(config) if not isinstance(config, dict) else config


def read_total_frames(data_path: str) -> int:
    with (Path(data_path) / 'meta' / 'info.json').open('r') as f:
        return int(json.load(f)['total_frames'])


def extract_groups(data_or_config: dict[str, Any]) -> tuple[list[list[int]], list[float], list[list[str]]]:
    if data_or_config.get('_class_name') != 'WeightedConcatDataset':
        raise ValueError('Expected top-level WeightedConcatDataset config.')

    lengths_by_group = []
    paths_by_group = []
    for group in data_or_config['datasets']:
        group_lengths = []
        group_paths = []
        for cfg in group:
            data_path = cfg['data_path']
            group_lengths.append(read_total_frames(data_path))
            group_paths.append(data_path)
        lengths_by_group.append(group_lengths)
        paths_by_group.append(group_paths)

    return lengths_by_group, list(data_or_config['sampling_weights']), paths_by_group


def summarize_lengths(lengths_by_group: list[list[int]]) -> dict[str, Any]:
    leaf_lengths = [length for group in lengths_by_group for length in group]
    group_lengths = [sum(group) for group in lengths_by_group]
    return {
        'groups': len(lengths_by_group),
        'leaf_datasets': len(leaf_lengths),
        'frames': sum(leaf_lengths),
        'hours_at_30fps': sum(leaf_lengths) / 30 / 3600,
        'max_group_leafs': max(len(group) for group in lengths_by_group),
        'mean_group_leafs': float(np.mean([len(group) for group in lengths_by_group])),
        'max_group_frames': max(group_lengths),
        'min_group_frames': min(group_lengths),
    }


class CountingLeafDataset:
    def __init__(self, length: int, stats: Stats) -> None:
        self.length = int(length)
        self.stats = stats
        self.opened = False

    def open(self) -> None:
        self.stats.leaf_open_calls += 1
        if not self.opened:
            self.stats.leaf_first_open_calls += 1
            self.opened = True

    def close(self) -> None:
        self.opened = False

    def reset(self) -> None:
        self.close()

    def filter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, int]:
        self.open()
        self.stats.leaf_getitem_calls += 1
        return {'index': int(index)}


class BisectConcatDataset:
    def __init__(self, datasets: list[Any]) -> None:
        self.datasets = datasets
        lengths = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum(lengths).tolist()
        self.transform = None

    def open(self) -> None:
        for dataset in self.datasets:
            dataset.open()

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def reset(self) -> None:
        for dataset in self.datasets:
            dataset.reset()

    def filter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def set_transform(self, transform: Any) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        if isinstance(index, (list, tuple)):
            data = [self._get_data(idx) for idx in index]
        else:
            data = self._get_data(index)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _get_data(self, index: int) -> Any:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_index == len(self.cumulative_sizes):
            raise IndexError(index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][index - previous_size]


class BisectWeightedConcatDataset(BisectConcatDataset):
    def __init__(self, datasets: list[Any], sampling_weights: list[float]) -> None:
        super().__init__(datasets)
        self.sampling_weights = [float(weight) for weight in sampling_weights]


def build_current_tree(lengths_by_group: list[list[int]], sampling_weights: list[float], stats: Stats):
    from giga_datasets import ConcatDataset, WeightedConcatDataset

    groups = [ConcatDataset([CountingLeafDataset(length, stats) for length in group]) for group in lengths_by_group]
    return WeightedConcatDataset(groups, sampling_weights=sampling_weights)


def build_bisect_tree(lengths_by_group: list[list[int]], sampling_weights: list[float], stats: Stats):
    groups = [BisectConcatDataset([CountingLeafDataset(length, stats) for length in group]) for group in lengths_by_group]
    return BisectWeightedConcatDataset(groups, sampling_weights=sampling_weights)


def make_list_weighted_indices(dataset: Any, data_config: dict[str, Any], samples: int) -> tuple[list[int], float]:
    from giga_datasets import ListWeightedSampler

    sampler_cfg = dict(data_config.get('sampler', {'type': 'ListWeightedSampler'}))
    sampler_cfg.pop('type', None)
    sampler_cfg.setdefault('shuffle', True)
    sampler_cfg.pop('index_mode', None)
    sampler_cfg.pop('chunk_size', None)
    sampler_cfg.pop('shuffle_within_chunk', None)
    sampler_cfg['epoch_size'] = samples

    batch_size = int(data_config.get('batch_size_per_gpu', 1))
    tic = time.perf_counter()
    sampler = ListWeightedSampler(dataset=dataset, batch_size=batch_size, **sampler_cfg)
    indices = list(islice(iter(sampler), samples))
    return indices, time.perf_counter() - tic


def make_default_simulated_indices(lengths_by_group: list[list[int]], samples: int) -> tuple[list[int], float]:
    # DefaultSampler builds a full random permutation. For a small prefix from a
    # 50M-300M item dataset, random draws with replacement are an accurate and
    # memory-safe proxy for routing/order effects.
    total_size = sum(length for group in lengths_by_group for length in group)
    tic = time.perf_counter()
    rng = np.random.default_rng(6666)
    indices = rng.integers(0, total_size, size=samples, dtype=np.int64).tolist()
    return indices, time.perf_counter() - tic


def make_default_actual_indices(dataset: Any, data_config: dict[str, Any], samples: int) -> tuple[list[int], float]:
    from giga_datasets import DefaultSampler

    batch_size = int(data_config.get('batch_size_per_gpu', 1))
    shuffle = bool(data_config.get('sampler', {}).get('shuffle', True))
    tic = time.perf_counter()
    sampler = DefaultSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    indices = list(islice(iter(sampler), samples))
    return indices, time.perf_counter() - tic


def scale_lengths(lengths_by_group: list[list[int]], total_frames: int) -> list[list[int]]:
    leaf_lengths = [length for group in lengths_by_group for length in group]
    original_total = sum(leaf_lengths)
    if total_frames <= 0:
        raise ValueError('total_frames should be positive')
    if total_frames < len(leaf_lengths):
        raise ValueError('total_frames should be at least the number of leaf datasets')

    scaled_leaf_lengths = [max(1, int(round(length * total_frames / original_total))) for length in leaf_lengths]
    delta = total_frames - sum(scaled_leaf_lengths)
    if delta != 0:
        order = sorted(range(len(scaled_leaf_lengths)), key=lambda i: leaf_lengths[i], reverse=(delta > 0))
        step = 1 if delta > 0 else -1
        remaining = abs(delta)
        cursor = 0
        while remaining > 0:
            i = order[cursor % len(order)]
            if step > 0 or scaled_leaf_lengths[i] > 1:
                scaled_leaf_lengths[i] += step
                remaining -= 1
            cursor += 1

    scaled = []
    cursor = 0
    for group in lengths_by_group:
        scaled.append(scaled_leaf_lengths[cursor : cursor + len(group)])
        cursor += len(group)
    return scaled


def time_dataset_reads(dataset: Any, indices: list[int]) -> tuple[float, Stats]:
    stats = Stats()

    # Swap fresh stats into leaves so sampler construction does not count.
    def replace_stats(node: Any) -> None:
        if isinstance(node, CountingLeafDataset):
            node.stats = stats
            node.opened = False
            return
        for child in getattr(node, 'datasets', []):
            replace_stats(child)

    replace_stats(dataset)
    tic = time.perf_counter()
    checksum = 0
    for index in indices:
        checksum += dataset[index]['index']
    elapsed = time.perf_counter() - tic
    if checksum < 0:
        raise RuntimeError('unreachable')
    return elapsed, stats


def analytic_linear_steps(lengths_by_group: list[list[int]], indices: list[int]) -> dict[str, float]:
    group_lengths = [sum(group) for group in lengths_by_group]
    group_cum = np.cumsum(group_lengths).tolist()
    inner_cums = [np.cumsum(group).tolist() for group in lengths_by_group]

    outer_steps = 0
    inner_steps = 0
    selected_group_leafs = 0
    total_leafs = sum(len(group) for group in lengths_by_group)

    for index in indices:
        group_index = bisect.bisect_right(group_cum, index)
        group_start = 0 if group_index == 0 else group_cum[group_index - 1]
        local_index = index - group_start
        leaf_index = bisect.bisect_right(inner_cums[group_index], local_index)

        outer_steps += group_index + 1
        inner_steps += leaf_index + 1
        selected_group_leafs += len(lengths_by_group[group_index])

    n = len(indices)
    return {
        'outer_linear_steps_per_sample': outer_steps / n,
        'inner_linear_steps_per_sample': inner_steps / n,
        'open_leaf_calls_current_per_sample': total_leafs + selected_group_leafs / n + 1,
        'open_leaf_calls_bisect_per_sample': 1,
    }


def run_one(config_path: str, samples: int, sampler: str, scale_total_frames: int | None) -> None:
    print(f'\n=== {config_path} ===')
    config = load_experiment_config(config_path)
    data_config = config['dataloaders']['train']
    lengths_by_group, sampling_weights, _paths_by_group = extract_groups(data_config['data_or_config'])
    summary = summarize_lengths(lengths_by_group)
    print(
        'shape: '
        f"groups={summary['groups']}, leaf_datasets={summary['leaf_datasets']}, "
        f"frames={summary['frames']:,}, hours@30fps={summary['hours_at_30fps']:.2f}, "
        f"mean_group_leafs={summary['mean_group_leafs']:.2f}, max_group_leafs={summary['max_group_leafs']}"
    )

    benchmark_lengths = lengths_by_group
    if sampler == 'default-actual':
        if scale_total_frames is None:
            raise ValueError('--scale-total-frames is required for sampler=default-actual')
        benchmark_lengths = scale_lengths(lengths_by_group, scale_total_frames)
        print(f'default-actual uses scaled lengths: total_frames={sum(sum(g) for g in benchmark_lengths):,}')

    sampler_tree = build_current_tree(benchmark_lengths, sampling_weights, Stats())
    if sampler == 'list-weighted':
        indices, sampler_elapsed = make_list_weighted_indices(sampler_tree, data_config, samples)
        sampler_label = 'ListWeightedSampler'
    elif sampler == 'default-simulated':
        indices, sampler_elapsed = make_default_simulated_indices(benchmark_lengths, samples)
        sampler_label = 'DefaultSampler simulated random prefix'
    elif sampler == 'default-actual':
        indices, sampler_elapsed = make_default_actual_indices(sampler_tree, data_config, samples)
        sampler_label = 'DefaultSampler actual on scaled lengths'
    else:
        raise ValueError(f'Unknown sampler: {sampler}')

    print(
        f'sampler first {len(indices):,} indices: {sampler_elapsed:.3f}s, '
        f'{len(indices) / sampler_elapsed:,.0f} idx/s, rss={rss_gb():.2f}GB, sampler={sampler_label}'
    )

    analytic = analytic_linear_steps(benchmark_lengths, indices)
    print(
        'analytic current path: '
        f"outer_steps/sample={analytic['outer_linear_steps_per_sample']:.2f}, "
        f"inner_steps/sample={analytic['inner_linear_steps_per_sample']:.2f}, "
        f"leaf_open_calls/sample={analytic['open_leaf_calls_current_per_sample']:.2f}"
    )

    current_tree = build_current_tree(benchmark_lengths, sampling_weights, Stats())
    current_elapsed, current_stats = time_dataset_reads(current_tree, indices)
    print(
        f'current ConcatDataset routing: {current_elapsed:.3f}s, '
        f'{len(indices) / current_elapsed:,.0f} samples/s, '
        f'leaf_open_calls={current_stats.leaf_open_calls:,}, '
        f'leaf_first_opens={current_stats.leaf_first_open_calls:,}'
    )

    bisect_tree = build_bisect_tree(benchmark_lengths, sampling_weights, Stats())
    bisect_elapsed, bisect_stats = time_dataset_reads(bisect_tree, indices)
    print(
        f'bisect lazy routing: {bisect_elapsed:.3f}s, '
        f'{len(indices) / bisect_elapsed:,.0f} samples/s, '
        f'leaf_open_calls={bisect_stats.leaf_open_calls:,}, '
        f'leaf_first_opens={bisect_stats.leaf_first_open_calls:,}, '
        f'speedup={current_elapsed / bisect_elapsed:.2f}x'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--samples', type=int, default=100_000)
    parser.add_argument('--sampler', choices=['list-weighted', 'default-simulated', 'default-actual'], default='list-weighted')
    parser.add_argument('--scale-total-frames', type=int, default=None)
    args = parser.parse_args()

    ensure_pythonpaths()
    print(f'python={sys.executable}')
    print(f'samples={args.samples:,}, sampler={args.sampler}')
    for config_path in args.configs:
        run_one(config_path, args.samples, args.sampler, args.scale_total_frames)


if __name__ == '__main__':
    main()
