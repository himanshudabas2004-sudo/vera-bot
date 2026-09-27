#!/usr/bin/env python3
"""
generate_submission.py

Generates submission.jsonl for magicpin AI Challenge:
1. Loads test pairs from submission_test_pairs.json (30 pairs).
2. For each pair, loads merchant, category, trigger, and optional customer from expanded/ and dataset/categories/.
3. Directly invokes compose() from bot.py.
4. Writes submission.jsonl formatted as:
   {"test_id": "T01", "body": "...", "cta": "...", "send_as": "...", "suppression_key": "...", "rationale": "..."}
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_submission")

# Ensure bot can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bot import compose


async def generate_submission(
    pairs_file: Path = Path("submission_test_pairs.json"),
    output_file: Path = Path("submission.jsonl"),
    expanded_dir: Path = Path("expanded"),
    categories_dir: Path = Path("dataset/categories"),
) -> None:
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")

    with open(pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        pairs = data.get("pairs", [])

    logger.info(f"Loaded {len(pairs)} test pairs from {pairs_file}")

    records = []
    for pair in pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        # Load merchant
        merchant_path = expanded_dir / "merchants" / f"{merchant_id}.json"
        if not merchant_path.exists():
            raise FileNotFoundError(f"Merchant file not found: {merchant_path}")
        with open(merchant_path, "r", encoding="utf-8") as f:
            merchant = json.load(f)

        # Load category
        category_slug = merchant.get("category_slug", "general")
        category_path = categories_dir / f"{category_slug}.json"
        if not category_path.exists():
            category_path = expanded_dir / "categories" / f"{category_slug}.json"
        if not category_path.exists():
            raise FileNotFoundError(f"Category file not found for slug '{category_slug}'")
        with open(category_path, "r", encoding="utf-8") as f:
            category = json.load(f)

        # Load trigger
        trigger_path = expanded_dir / "triggers" / f"{trigger_id}.json"
        if not trigger_path.exists():
            raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
        with open(trigger_path, "r", encoding="utf-8") as f:
            trigger = json.load(f)

        # Load customer (optional)
        customer = None
        if customer_id:
            customer_path = expanded_dir / "customers" / f"{customer_id}.json"
            if not customer_path.exists():
                raise FileNotFoundError(f"Customer file not found: {customer_path}")
            with open(customer_path, "r", encoding="utf-8") as f:
                customer = json.load(f)

        # Stable conversation ID
        conv_id = f"conv_{test_id}_{merchant_id}_{trigger_id}"

        # Call compose directly
        result = await compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            conversation_id=conv_id,
        )

        record = {
            "test_id": test_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "binary_yes_stop"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", f"suppress:{merchant_id}:{trigger_id}"),
            "rationale": result.get("rationale", ""),
        }
        records.append(record)
        logger.info(f"[{test_id}] Generated ({record['send_as']}): {record['body'][:60]}...")

    # Write submission.jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated {len(records)} lines in {output_file}")


def main() -> None:
    asyncio.run(generate_submission())


if __name__ == "__main__":
    main()
