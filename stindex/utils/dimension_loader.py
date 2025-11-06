"""
Dimension configuration loader and utilities.

Loads dimension definitions from YAML files and provides utilities
for working with multi-dimensional extraction configs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from stindex.llm.response.dimension_models import DimensionMetadata, DimensionType


class DimensionConfig(BaseModel):
    """Validated dimension configuration."""

    name: str
    enabled: bool = True
    description: str
    extraction_type: str  # normalized, geocoded, categorical, structured, free_text
    schema_type: str  # Name of the schema class

    # Field definitions
    fields: List[Dict[str, Any]] = Field(default_factory=list)

    # Examples for few-shot learning
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    # Type-specific configuration
    normalization: Optional[Dict[str, Any]] = None
    disambiguation: Optional[Dict[str, Any]] = None
    geocoding: Optional[Dict[str, Any]] = None

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
    """Loads and manages dimension configurations."""

    def __init__(self, config_dir: str = "cfg"):
        """
        Initialize loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)

    def load_dimension_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load dimension configuration from YAML file.

        Args:
            config_path: Path to config file (relative to config_dir or absolute)
                        Can be:
                        - "dimensions" → cfg/dimensions.yml
                        - "case_studies/public_health/extraction/config/health_dimensions" → full path
                        - "/absolute/path/to/config.yml"

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

        return config

    def get_enabled_dimensions(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, DimensionConfig]:
        """
        Get all enabled dimensions from config.

        Args:
            config: Loaded config dict

        Returns:
            Dict of dimension name → DimensionConfig
        """
        dimensions = config.get("dimensions", {})
        enabled_dims = {}

        for dim_name, dim_config in dimensions.items():
            if dim_config.get("enabled", True):
                try:
                    dimension = DimensionConfig(
                        name=dim_name,
                        **dim_config
                    )
                    enabled_dims[dim_name] = dimension
                except Exception as e:
                    logger.warning(f"Failed to parse dimension '{dim_name}': {e}")

        logger.info(f"Loaded {len(enabled_dims)} enabled dimensions: {list(enabled_dims.keys())}")
        return enabled_dims

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

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    # Merge top-level keys
    for key, value in override_config.items():
        if key == "dimensions" and key in merged:
            # Merge dimensions dict
            merged["dimensions"] = {
                **merged.get("dimensions", {}),
                **value
            }
        else:
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
        required_dim_keys = ["enabled", "description", "extraction_type", "schema_type", "fields"]
        for key in required_dim_keys:
            if key not in dim_config:
                raise ValueError(f"Dimension '{dim_name}' missing required key: {key}")

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

    # Build JSON schema
    schema = loader.build_json_schema(dimensions)
    print("\nJSON Schema:")
    print(json.dumps(schema, indent=2))
