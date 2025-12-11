#!/usr/bin/env python3
"""
Dimension Configuration Migration Script

Migrates dimension configs from old field-based format to new hierarchy-based format.

Old Format (fields-based):
    temporal:
      fields:
        - name: text
          type: string
        - name: normalized
          type: string
        - name: normalization_type
          type: enum
          values: [...]

New Format (hierarchy-based):
    temporal:
      hierarchy:
        - level: timestamp
          description: "Specific date and time (ISO 8601)"
        - level: date
          description: "Calendar date (YYYY-MM-DD)"
        - level: month
          description: "Year and month (YYYY-MM)"
        - level: year
          description: "Calendar year (YYYY)"

Usage:
    # Migrate base config
    python scripts/migrate_dimension_configs.py \\
        --input cfg/extraction/inference/dimensions.yml \\
        --output cfg/extraction/inference/dimensions.yml \\
        --backup

    # Migrate case study config (with base merging)
    python scripts/migrate_dimension_configs.py \\
        --input case_studies/public_health/config/health_dimensions.yml \\
        --output case_studies/public_health/config/health_dimensions.yml \\
        --base cfg/extraction/inference/dimensions.yml \\
        --backup
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Predefined mandatory hierarchies (from dimension_loader.py)
MANDATORY_TEMPORAL_HIERARCHY = [
    {"level": "timestamp", "description": "Specific date and time (ISO 8601)"},
    {"level": "date", "description": "Calendar date (YYYY-MM-DD)"},
    {"level": "month", "description": "Year and month (YYYY-MM)"},
    {"level": "year", "description": "Calendar year (YYYY)"}
]

MANDATORY_SPATIAL_HIERARCHY = [
    {"level": "location", "description": "Specific location name, address, or point of interest"},
    {"level": "city", "description": "City or municipality"},
    {"level": "state", "description": "State, province, or region"},
    {"level": "country", "description": "Country name"}
]


def migrate_dimension(dim_name: str, dim_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a single dimension from old format to new format.

    Args:
        dim_name: Dimension name
        dim_config: Dimension configuration dict

    Returns:
        Migrated dimension config
    """
    migrated = dim_config.copy()

    # Check if already in new format (has hierarchy)
    if "hierarchy" in migrated:
        print(f"  ✓ {dim_name}: Already in hierarchy format, skipping")
        return migrated

    # Check if has old format (has fields)
    if "fields" not in migrated:
        print(f"  ⚠ {dim_name}: No fields or hierarchy found, skipping")
        return migrated

    print(f"  → Migrating {dim_name}...")

    # Remove fields section
    fields = migrated.pop("fields", [])

    # Handle temporal dimension (use predefined hierarchy)
    if dim_name == "temporal":
        migrated["hierarchy"] = MANDATORY_TEMPORAL_HIERARCHY
        # Update schema_type to TemporalEntity
        migrated["schema_type"] = "TemporalEntity"
        print(f"    • Applied predefined temporal hierarchy (4 levels)")

    # Handle spatial dimension (use predefined hierarchy)
    elif dim_name == "spatial":
        migrated["hierarchy"] = MANDATORY_SPATIAL_HIERARCHY
        # Update schema_type to SpatialEntity
        migrated["schema_type"] = "SpatialEntity"
        print(f"    • Applied predefined spatial hierarchy (4 levels)")

    # Handle other dimensions (convert from enum fields)
    else:
        hierarchy = _convert_fields_to_hierarchy(fields, dim_name)
        if hierarchy:
            migrated["hierarchy"] = hierarchy
            # Update schema_type to GenericEntity
            migrated["schema_type"] = "GenericEntity"
            print(f"    • Generated hierarchy with {len(hierarchy)} levels")
        else:
            print(f"    ⚠ Could not generate hierarchy, keeping original config")
            migrated["fields"] = fields  # Restore fields if migration failed

    return migrated


def _convert_fields_to_hierarchy(fields: List[Dict[str, Any]], dim_name: str) -> List[Dict[str, Any]]:
    """
    Convert fields list to hierarchy levels.

    Args:
        fields: List of field definitions
        dim_name: Dimension name

    Returns:
        List of hierarchy level definitions
    """
    hierarchy = []

    for field in fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "string")
        field_desc = field.get("description", "")

        # Skip "text" field (auto-added by DimensionLoader)
        if field_name == "text":
            continue

        # Skip "confidence" field (not part of hierarchy)
        if field_name == "confidence":
            continue

        # For normalized dimensions, skip "normalized" field
        if field_name == "normalized":
            continue

        # Convert field to hierarchy level
        level_def = {
            "level": field_name,
            "description": field_desc or f"{field_name.replace('_', ' ').title()} level"
        }

        # If enum field, add values
        if field_type == "enum" and "values" in field:
            level_def["values"] = field["values"]

        hierarchy.append(level_def)

    return hierarchy


def migrate_config(
    input_path: Path,
    output_path: Path,
    base_config: Optional[Dict[str, Any]] = None,
    backup: bool = True
) -> bool:
    """
    Migrate a dimension config file.

    Args:
        input_path: Input config path
        output_path: Output config path
        base_config: Optional base config to merge with
        backup: Create backup before overwriting

    Returns:
        True if successful
    """
    print(f"\n{'=' * 80}")
    print(f"Migrating: {input_path}")
    print(f"{'=' * 80}\n")

    # Load input config
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return False

    with open(input_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge with base config if provided
    if base_config:
        print(f"Merging with base config...")
        config = merge_configs(base_config, config)

    # Create backup
    if backup and output_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = output_path.with_suffix(f'.{timestamp}.backup.yml')
        shutil.copy(output_path, backup_path)
        print(f"✓ Backup created: {backup_path}\n")

    # Migrate dimensions
    if "dimensions" not in config:
        print("⚠ No dimensions found in config")
        return False

    dimensions = config["dimensions"]
    migrated_dimensions = {}

    for dim_name, dim_config in dimensions.items():
        migrated_dim = migrate_dimension(dim_name, dim_config)
        migrated_dimensions[dim_name] = migrated_dim

    config["dimensions"] = migrated_dimensions

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"\n✅ Migration complete!")
    print(f"   Output: {output_path}")
    print(f"   Migrated {len(migrated_dimensions)} dimensions")

    return True


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configs (override takes precedence).

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if key == "dimensions" and key in merged:
            # Deep merge dimensions
            for dim_name, dim_config in value.items():
                if dim_name in merged["dimensions"]:
                    # Merge this dimension's config
                    merged_dim = copy.deepcopy(merged["dimensions"][dim_name])
                    for dim_key, dim_value in dim_config.items():
                        merged_dim[dim_key] = dim_value
                    merged["dimensions"][dim_name] = merged_dim
                else:
                    # New dimension, add it
                    merged["dimensions"][dim_name] = dim_config
        elif isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Deep merge nested dicts
            merged[key] = {**merged.get(key, {}), **value}
        else:
            # Override scalar values and lists
            merged[key] = value

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Migrate dimension configs from field-based to hierarchy-based format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='Input config file path'
    )
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Output config file path'
    )
    parser.add_argument(
        '--base',
        type=Path,
        help='Base config to merge with (for case study configs)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='Create backup before overwriting (default: True)'
    )
    parser.add_argument(
        '--no-backup',
        dest='backup',
        action='store_false',
        help='Skip backup creation'
    )

    args = parser.parse_args()

    # Load base config if provided
    base_config = None
    if args.base:
        if not args.base.exists():
            print(f"✗ Base config not found: {args.base}")
            return 1

        with open(args.base, 'r') as f:
            base_config = yaml.safe_load(f)
        print(f"Loaded base config: {args.base}")

    # Migrate
    success = migrate_config(
        input_path=args.input,
        output_path=args.output,
        base_config=base_config,
        backup=args.backup
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
