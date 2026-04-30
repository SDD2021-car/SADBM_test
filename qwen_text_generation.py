#!/usr/bin/env python3
"""批量调用千问视觉模型：按用户问题为每张图生成描述，并保存为 JSON。"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量调用千问视觉 API，为图片按问题生成描述并导出 JSON"
    )
    parser.add_argument("--image-dir", default="/data/hjf/Dataset/SEN12_Scene/testB")
    parser.add_argument("--question", default = "请用中文描述这张遥感图像的场景内容，包括分辨率、季节、地貌类型和植被特征,输入图片不包含冬季的图片，30字以内")
    parser.add_argument("--output", default="captions_test_scene_no_class.json", help="输出 JSON 路径")
    parser.add_argument(
        "--api-key",
        default="sk-ca89ee6705e34f1bbb6f17baabc4c55d",
        help="千问 API Key，默认读取 DASHSCOPE_API_KEY",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="千问兼容 OpenAI API 的 base URL",
    )
    parser.add_argument("--model", default="qwen2.5-vl-32b-instruct", help="模型名")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--max-retries", type=int, default=3, help="单图失败重试次数")
    parser.add_argument(
        "--retry-failed-only",
        default=True,
        help="仅重试已有输出 JSON 中 error 不为空的记录",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每处理多少张做一次落盘（防止中途失败丢失）",
    )
    return parser.parse_args()


def encode_image_to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg"
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def ask_one_image(
    client: OpenAI,
    model: str,
    image_path: Path,
    question: str,
    max_retries: int,
) -> dict[str, Any]:
    data_uri = encode_image_to_data_uri(image_path)

    last_err = None
    for i in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            )
            content = resp.choices[0].message.content
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                answer = "\n".join(t for t in text_parts if t).strip()
            else:
                answer = (content or "").strip()

            return {
                "image": str(image_path),
                "question": question,
                "answer": answer,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if i < max_retries:
                time.sleep(min(2**i, 8))

    return {
        "image": str(image_path),
        "question": question,
        "answer": None,
        "error": last_err,
    }


def save_json(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")



def load_existing_results(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception as exc:  # noqa: BLE001
        print(f"警告: 读取已有 JSON 失败，将从空结果开始。错误: {exc}", flush=True)
    return []


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("未提供 API Key，请传 --api-key 或设置 DASHSCOPE_API_KEY")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise SystemExit(f"图片目录不存在: {image_dir}")

    all_image_paths = sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not all_image_paths:
        raise SystemExit(f"未在 {image_dir} 中找到支持的图片格式: {sorted(SUPPORTED_EXTS)}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    output_path = Path(args.output)
    results = load_existing_results(output_path)

    if args.retry_failed_only:
        failed_images = {
            row.get("image")
            for row in results
            if row.get("error") and isinstance(row.get("image"), str)
        }
        image_paths = [p for p in all_image_paths if str(p) in failed_images]
        print(
            f"检测到已有记录 {len(results)} 条，其中失败图片 {len(image_paths)} 张，开始重试...",
            flush=True,
        )
    else:
        image_paths = all_image_paths

    if not image_paths:
        print("没有需要处理的图片，任务结束。", flush=True)
        return

    print(f"本次待处理 {len(image_paths)} 张图片，开始处理...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                ask_one_image,
                client,
                args.model,
                img_path,
                args.question,
                args.max_retries,
            ): img_path
            for img_path in image_paths
        }

        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()

            # 用新结果替换旧记录；如果原来没有，则追加
            replaced = False
            for i, old in enumerate(results):
                if old.get("image") == result["image"]:
                    results[i] = result
                    replaced = True
                    break

            if not replaced:
                results.append(result)
            ok = result["error"] is None
            status = "OK" if ok else "ERR"
            print(f"[{idx}/{len(image_paths)}] {status} - {result['image']}", flush=True)

            if idx % args.save_every == 0:
                save_json(output_path, results)

    results = sorted(results, key=lambda x: x["image"])
    save_json(Path(args.output), results)

    success = sum(1 for r in results if r["error"] is None)
    failed = len(results) - success
    print(f"完成: 当前 JSON 累计成功 {success} 条, 失败 {failed} 条 -> {args.output}")


if __name__ == "__main__":
    main()



