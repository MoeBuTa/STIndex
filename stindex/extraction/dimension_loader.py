"""
Dimension configuration loader and utilities.

Loads dimension definitions from YAML files and provides utilities
for working with multi-dimensional extraction configs.

Supports hierarchy-based dimension schemas for discovered dimensions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from stindex.llm.response.dimension_models import DimensionMetadata, DimensionType


# Predefined mandatory dimensions with hierarchies
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


class DimensionConfig(BaseModel):
    """Validated dimension configuration."""

    name: str
    enabled: bool = True
    description: str
    extraction_type: str  # normalized, geocoded, categorical, structured, free_text
    schema_type: str = "GenericEntity"  # Name of the schema class

    # Field definitions (legacy format OR auto-generated from hierarchy)
    fields: List[Dict[str, Any]] = Field(default_factory=list)

    # Hierarchy levels (new format - takes precedence over fields if present)
    hierarchy: Optional[List[Dict[str, Any]]] = None

    # Examples for few-shot learning
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    # Type-specific configuration
    normalization: Optional[Dict[str, Any]] = None
    disambiguation: Optional[Dict[str, Any]] = None
    geocoding: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization: auto-generate fields from hierarchy if present."""
        if self.hierarchy and not self.fields:
            # Auto-generate fields from hierarchy
            self.fields = self._generate_fields_from_hierarchy()
            logger.debug(f"Auto-generated {len(self.fields)} fields from hierarchy for dimension '{self.name}'")

    def _generate_fields_from_hierarchy(self) -> List[Dict[str, Any]]:
        """
        Generate field definitions from hierarchy levels.

        Returns:
            List of field definitions
        """
        fields = []

        # Always add 'text' field first
        fields.append({
            "name": "text",
            "type": "string",
            "description": "Original text mention from document",
            "required": True
        })

        # Add hierarchy level fields
        for level_def in self.hierarchy:
            level_name = level_def.get("level")
            level_desc = level_def.get("description", f"{level_name} level")
            level_values = level_def.get("values", [])

            if level_values:
                # If values provided, make it an enum field
                fields.append({
                    "name": level_name,
                    "type": "enum",
                    "values": level_values,
                    "description": level_desc,
                    "required": False  # Hierarchy levels are optional
                })
            else:
                # Free-text field
                fields.append({
                    "name": level_name,
                    "type": "string",
                    "description": level_desc,
                    "required": False
                })

        return fields

    def to_metadata(self) -> DimensionMetadata:
        """Convert to DimensionMetadata."""
        return DimensionMetadata(
            name=self.name,
            enabled=self.enabled,
            description=self.description,
            extraction_type=DimensionType(self.extraction_type),
            schema_type=self.schema_type,
            fields=self.fields,
            examples=self.examples
        )


class DimensionConfigLoader:
    """Loads and manages dimension configurations with hierarchy support."""

    def __init__(self, config_dir: str = "cfg/extraction/inference"):
        """
        Initialize loader.

        Args:
            config_dir: Directory containing config files (default: cfg/extraction/inference)
        """
        self.config_dir = Path(config_dir)

    def load_dimension_config(self, config_path: str, auto_merge_base: bool = True) -> Dict[str, Any]:
        """
        Load dimension configuration from YAML file.

        Args:
            config_path: Path to config file (relative to config_dir or absolute)
                        Can be:
                        - "dimensions" → cfg/extraction/inference/dimensions.yml
                        - "case_studies/public_health/config/health_dimensions" → full path
                        - "/absolute/path/to/config.yml"
            auto_merge_base: If True and loading a non-base config, automatically merge
                           with base cfg/dimensions.yml (default: True)

        Returns:
            Parsed configuration dict
        """
        # Handle different path formats
        if not config_path.endswith('.yml') and not config_path.endswith('.yaml'):
            config_path = f"{config_path}.yml"

        config_file = Path(config_path)

        # If not absolute, try relative to config_dir
        if not config_file.is_absolute():
            config_file = self.config_dir / config_file

        # If still not found, try relative to project root
        if not config_file.exists():
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Dimension config not found: {config_path}")

        logger.info(f"Loading dimension config from: {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Auto-merge with base config if this is not the base config itself
        if auto_merge_base and not config_path.startswith("dimensions"):
            try:
                # Try cfg/extraction/inference/dimensions.yml first, then cfg/dimensions.yml
                base_config_file = Path("cfg/extraction/inference") / "dimensions.yml"
                if not base_config_file.exists():
                    base_config_file = Path("cfg") / "dimensions.yml"

                if base_config_file.exists():
                    logger.info(f"  Merging with base config: {base_config_file}")
                    with open(base_config_file, 'r') as f:
                        base_config = yaml.safe_load(f)
                    config = merge_dimension_configs(base_config, config)
                    logger.info("  ✓ Config merged with base dimensions")
            except Exception as e:
                logger.warning(f"Failed to merge with base config: {e}")

        return config

    def get_enabled_dimensions(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, DimensionConfig]:
        """
        Get all enabled dimensions from config.

        Ensures mandatory temporal and spatial dimensions are always present.

        Args:
            config: Loaded config dict

        Returns:
            Dict of dimension name → DimensionConfig
        """
        dimensions = config.get("dimensions", {})
        enabled_dims = {}

        # First, load explicitly configured dimensions
        for dim_name, dim_config in dimensions.items():
            if dim_config.get("enabled", True):
                try:
                    # Infer extraction_type if not specified
                    if "extraction_type" not in dim_config:
                        dim_config["extraction_type"] = self._infer_extraction_type(dim_name, dim_config)

                    # Set default schema_type if not specified
                    if "schema_type" not in dim_config:
                        dim_config["schema_type"] = "GenericEntity"

                    dimension = DimensionConfig(
                        name=dim_name,
                        **dim_config
                    )
                    enabled_dims[dim_name] = dimension
                except Exception as e:
                    logger.warning(f"Failed to parse dimension '{dim_name}': {e}")

        # Ensure mandatory dimensions are present
        enabled_dims = self._ensure_mandatory_dimensions(enabled_dims)

        logger.info(f"Loaded {len(enabled_dims)} enabled dimensions: {list(enabled_dims.keys())}")
        return enabled_dims

    def _infer_extraction_type(self, dim_name: str, dim_config: Dict[str, Any]) -> str:
        """
        Infer extraction type from dimension name and config.

        Args:
            dim_name: Dimension name
            dim_config: Dimension configuration

        Returns:
            Inferred extraction type
        """
        # Check for explicit indicators
        if dim_name == "temporal":
            return "normalized"
        elif dim_name == "spatial":
            return "geocoded"

        # Check hierarchy for lat/lon (spatial)
        hierarchy = dim_config.get("hierarchy", [])
        if hierarchy:
            hierarchy_levels = [h.get("level") for h in hierarchy if isinstance(h, dict)]
            if any(level in ["latitude", "longitude", "lat", "lon"] for level in hierarchy_levels):
                return "geocoded"

        # Default to categorical for hierarchical dimensions
        return "categorical"

    def _ensure_mandatory_dimensions(
        self,
        dimensions: Dict[str, DimensionConfig]
    ) -> Dict[str, DimensionConfig]:
        """
        Ensure temporal and spatial dimensions are present with predefined hierarchies.

        Args:
            dimensions: Current dimension dict

        Returns:
            Dimension dict with mandatory dimensions added if missing
        """
        # Add temporal if missing
        if "temporal" not in dimensions:
            logger.info("Adding mandatory temporal dimension with predefined hierarchy")
            dimensions["temporal"] = DimensionConfig(
                name="temporal",
                enabled=True,
                description="Extract temporal expressions (dates, times, timestamps)",
                extraction_type="normalized",
                schema_type="TemporalEntity",
                hierarchy=MANDATORY_TEMPORAL_HIERARCHY,
                fields=[]  # Will be auto-generated from hierarchy
            )

        # Add spatial if missing
        if "spatial" not in dimensions:
            logger.info("Adding mandatory spatial dimension with predefined hierarchy")
            dimensions["spatial"] = DimensionConfig(
                name="spatial",
                enabled=True,
                description="Extract spatial/location mentions",
                extraction_type="geocoded",
                schema_type="SpatialEntity",
                hierarchy=MANDATORY_SPATIAL_HIERARCHY,
                fields=[]  # Will be auto-generated from hierarchy
            )

        return dimensions

    def build_json_schema(
        self,
        dimensions: Dict[str, DimensionConfig]
    ) -> Dict[str, Any]:
        """
        Build JSON schema for extraction result with all dimensions.

        Args:
            dimensions: Dict of dimension name → DimensionConfig

        Returns:
            JSON schema dict
        """
        properties = {}

        for dim_name, dim_config in dimensions.items():
            # Build field schema
            field_schema = self._build_dimension_field_schema(dim_config)
            properties[dim_name] = {
                "type": "array",
                "items": field_schema,
                "description": dim_config.description
            }

        schema = {
            "type": "object",
            "properties": properties,
            "required": []  # No required fields by default
        }

        return schema

    def _build_dimension_field_schema(
        self,
        dimension: DimensionConfig
    ) -> Dict[str, Any]:
        """
        Build JSON schema for a single dimension's field structure.

        Args:
            dimension: DimensionConfig

        Returns:
            JSON schema for the dimension's items
        """
        properties = {}
        required = []

        for field in dimension.fields:
            field_name = field.get("name")
            field_type = field.get("type")
            field_desc = field.get("description", "")
            field_required = field.get("required", True)

            if field_required:
                required.append(field_name)

            # Map field type to JSON schema type
            if field_type == "string":
                properties[field_name] = {"type": "string", "description": field_desc}
            elif field_type == "float":
                properties[field_name] = {"type": "number", "description": field_desc}
            elif field_type == "int":
                properties[field_name] = {"type": "integer", "description": field_desc}
            elif field_type == "enum":
                values = field.get("values", [])
                properties[field_name] = {
                    "type": "string",
                    "enum": values,
                    "description": field_desc
                }
            else:
                properties[field_name] = {"type": "string", "description": field_desc}

        schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        return schema

    def get_dimension_examples(
        self,
        dimensions: Dict[str, DimensionConfig]
    ) -> List[Dict[str, Any]]:
        """
        Get examples for few-shot learning from dimension configs.

        Args:
            dimensions: Dict of dimension name → DimensionConfig

        Returns:
            List of example dicts
        """
        all_examples = []

        for dim_name, dim_config in dimensions.items():
            for example in dim_config.examples:
                all_examples.append({
                    "dimension": dim_name,
                    **example
                })

        return all_examples

    def get_post_processing_rules(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Get post-processing rules from config.

        Args:
            config: Loaded config dict

        Returns:
            Dict of dimension type → list of processing steps
        """
        return config.get("post_processing", {})

    def get_linking_rules(
        self,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get cross-dimensional linking rules.

        Args:
            config: Loaded config dict

        Returns:
            List of linking rule dicts
        """
        linking_config = config.get("linking", {})
        if not linking_config.get("enabled", False):
            return []

        return linking_config.get("rules", [])

    def get_document_metadata_config(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get document metadata configuration.

        Args:
            config: Loaded config dict

        Returns:
            Document metadata config dict
        """
        return config.get("document_metadata", {})


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dimension_config(config_path: str = "dimensions") -> Dict[str, Any]:
    """
    Convenience function to load dimension config.

    Args:
        config_path: Path to config file

    Returns:
        Parsed config dict
    """
    loader = DimensionConfigLoader()
    return loader.load_dimension_config(config_path)


def get_dimension_schema(config_path: str = "dimensions") -> Dict[str, Any]:
    """
    Get JSON schema for dimensions from config.

    Args:
        config_path: Path to config file

    Returns:
        JSON schema dict
    """
    loader = DimensionConfigLoader()
    config = loader.load_dimension_config(config_path)
    dimensions = loader.get_enabled_dimensions(config)
    return loader.build_json_schema(dimensions)


def merge_dimension_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two dimension configs (override takes precedence).

    Deep merge strategy:
    - Top-level keys: override replaces base
    - dimensions: deep merge each dimension individually
      - If dimension exists in both: merge fields, override takes precedence
      - If dimension only in override: use as-is
      - If dimension only in base: use base

    Args:
        base_config: Base configuration (e.g., cfg/dimensions.yml)
        override_config: Override configuration (e.g., case study config)

    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)

    # Merge top-level keys (extraction, geocoding, document_metadata, etc.)
    for key, value in override_config.items():
        if key == "dimensions" and key in merged:
            # Deep merge dimensions
            base_dims = merged.get("dimensions", {})
            override_dims = value

            for dim_name, dim_config in override_dims.items():
                if dim_name in base_dims:
                    # Merge this dimension's config
                    merged_dim = copy.deepcopy(base_dims[dim_name])
                    for dim_key, dim_value in dim_config.items():
                        if isinstance(dim_value, dict) and dim_key in merged_dim:
                            # Deep merge nested dicts (e.g., normalization, disambiguation)
                            merged_dim[dim_key] = {**merged_dim.get(dim_key, {}), **dim_value}
                        else:
                            # Override scalar values
                            merged_dim[dim_key] = dim_value
                    merged["dimensions"][dim_name] = merged_dim
                else:
                    # New dimension in override, add it
                    merged["dimensions"][dim_name] = dim_config
        elif isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Deep merge nested dicts
            merged[key] = {**merged.get(key, {}), **value}
        else:
            # Override scalar values and lists
            merged[key] = value

    return merged


def validate_dimension_config(config: Dict[str, Any]) -> bool:
    """
    Validate dimension configuration structure.

    Args:
        config: Config dict to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["dimensions"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in dimension config: {key}")

    dimensions = config["dimensions"]
    if not isinstance(dimensions, dict):
        raise ValueError("'dimensions' must be a dict")

    for dim_name, dim_config in dimensions.items():
        # Check for either fields or hierarchy (new format)
        has_fields = "fields" in dim_config
        has_hierarchy = "hierarchy" in dim_config

        if not has_fields and not has_hierarchy:
            raise ValueError(f"Dimension '{dim_name}' must have either 'fields' or 'hierarchy'")

    return True


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Load and inspect dimension config
    loader = DimensionConfigLoader()

    # Load generic dimensions
    config = loader.load_dimension_config("dimensions")
    dimensions = loader.get_enabled_dimensions(config)

    print("Enabled dimensions:")
    for dim_name, dim_config in dimensions.items():
        print(f"  - {dim_name}: {dim_config.description}")
        if dim_config.hierarchy:
            hierarchy_levels = [h.get("level") for h in dim_config.hierarchy]
            print(f"    Hierarchy: {' → '.join(hierarchy_levels)}")

    # Build JSON schema
    schema = loader.build_json_schema(dimensions)
    print("\nJSON Schema:")
    print(json.dumps(schema, indent=2))
