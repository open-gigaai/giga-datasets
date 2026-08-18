import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from giga_datasets.datasets import LeRobotVQADataset


def _read_roots(args: argparse.Namespace) -> list[str]:
    roots: list[str] = []
    if args.list_file is not None:
        with open(args.list_file, 'r') as f:
            roots.extend(line.strip() for line in f if line.strip())
    roots.extend(args.roots)
    if args.max_roots is not None:
        roots = roots[: args.max_roots]
    if len(roots) == 0:
        raise ValueError('Pass at least one root path or --list_file.')
    return roots


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return ['']

    lines: list[str] = []
    current = words[0]
    probe = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(probe)

    for word in words[1:]:
        candidate = f'{current} {word}'
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _render_text_panel(image: Image.Image, sample: dict, *, panel_width: int | None = None) -> Image.Image:
    font = ImageFont.load_default()
    panel_width = max(panel_width or image.width, image.width)
    padding = 12
    line_gap = 4
    max_text_width = panel_width - 2 * padding
    text_lines = [
        f'sample_id: {sample["sample_id"]}',
        f'episode/frame/qa: {sample["episode_index"]}/{sample["frame_index"]}/{sample["qa_index"]}',
        f'Q: {sample["question"]}',
        f'A: {sample["answer"]}',
    ]

    wrapped_lines: list[str] = []
    for line in text_lines:
        wrapped_lines.extend(_wrap_text(line, font, max_text_width))

    line_heights = []
    probe = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(probe)
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    panel_height = padding * 2 + sum(line_heights) + line_gap * max(0, len(wrapped_lines) - 1)

    canvas = Image.new('RGB', (panel_width, image.height + panel_height), color=(255, 255, 255))
    canvas.paste(image, ((panel_width - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line, line_height in zip(wrapped_lines, line_heights):
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_height + line_gap
    return canvas


def _save_tensor_image(image: torch.Tensor, path: Path, sample: dict, *, overlay_text: bool = True) -> None:
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f'Expected image tensor with shape CxHxW or TxCxHxW, got {tuple(image.shape)}')
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_image = Image.fromarray(array)
    if overlay_text:
        pil_image = _render_text_panel(pil_image, sample)
    pil_image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Randomly inspect LeRobot VQA QA pairs and decoded frames.')
    parser.add_argument('roots', nargs='*', help='LeRobot root paths.')
    parser.add_argument('--list_file', help='Text file with one LeRobot root path per line.')
    parser.add_argument('--max_roots', type=int, help='Only inspect the first N roots from inputs.')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--output_dir', default='/tmp/lerobot_vqa_inspect')
    parser.add_argument('--tolerance_s', type=float, default=2e-2)
    parser.add_argument('--no_overlay_text', action='store_true', help='Save plain frames without QA text panel.')
    args = parser.parse_args()

    roots = _read_roots(args)
    dataset = LeRobotVQADataset(data_path=roots, tolerance_s=args.tolerance_s)
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise RuntimeError('LeRobotVQADataset has zero QA samples.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = rng.sample(range(dataset_size), k=min(args.num_samples, dataset_size))

    print(f'dataset_size={dataset_size}')
    print(f'output_dir={output_dir}')
    for rank, index in enumerate(indices):
        sample = dataset[index]
        image_path = output_dir / f'{rank:02d}_idx-{index}_sample-{sample["sample_id"].replace(":", "-")}.png'
        _save_tensor_image(sample['image'], image_path, sample, overlay_text=not args.no_overlay_text)

        print('=' * 80)
        print(f'index: {index}')
        print(f'sample_id: {sample["sample_id"]}')
        print(f'source_root: {sample["source_root"]}')
        print(f'episode_index: {sample["episode_index"]}')
        print(f'frame_index: {sample["frame_index"]}')
        print(f'camera: {sample["camera"]}')
        print(f'qa_index: {sample["qa_index"]}')
        print(f'question: {sample["question"]}')
        print(f'answer: {sample["answer"]}')
        print(f'image: {image_path}')


if __name__ == '__main__':
    main()
