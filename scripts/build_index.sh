#!/usr/bin/env bash
set -e
echo "Downloading dataset..."
python -m backend.ingestion.load_dataset

echo "Building vector index..."
python -m backend.ingestion.build_index

echo "Done. Index saved to backend/data/index"
