# Dataset Card: animal-image-detection:2026.07.01

## Overview

This curated release contains the spring baseline plus newly added summer image samples.

## Dataset Version

- Data asset name: animal-image-detection
- Version: 2026.07.01
- Label schema version: v1
- Release date: 2026-07-01

## Data Sources

| Source | Season | Count |
|---|---|---:|
| camera-a | spring | 4 |
| camera-b | summer | 3 |

## Splits

| Split | Count |
|---|---:|
| train | 3 |
| val | 2 |
| test | 2 |

## Label Distribution

| Label | Count |
|---|---:|
| cat | 4 |
| dog | 3 |

## Exclusions

| Reason | Count |
|---|---:|
| duplicate | 1 |
| low_quality | 1 |

## Known Limitations

- Autumn and winter images are not included in this version.
- This repository uses tiny PPM files for local smoke tests; production data should point to real image files in Azure Storage.