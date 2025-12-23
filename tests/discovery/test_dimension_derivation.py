"""
Unit tests for dimension derivation from extracted entities.

Tests the derive_dimensions_from_entities() utility function that derives
dimension schemas by aggregating unique dimension values and inferring
hierarchy structure from entity fields.
"""

import pytest
from pydantic import ValidationError

from stindex.discovery.models import (
    derive_dimensions_from_entities,
    DiscoveredDimensionSchema
)


class TestDeriveDimensionsFromEntities:
    """Tests for derive_dimensions_from_entities function."""

    def test_derive_single_dimension(self):
        """Test deriving single dimension from entities."""
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",
                "item_category": "fruit"
            },
            "banana": {
                "dimension": "Product",
                "specific_item": "banana",
                "item_category": "fruit"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        assert "Product" in derived
        assert len(derived) == 1
        assert "specific_item" in derived["Product"].hierarchy
        assert "item_category" in derived["Product"].hierarchy
        assert "apple" in derived["Product"].examples or "banana" in derived["Product"].examples

    def test_derive_multiple_dimensions(self):
        """Test deriving multiple dimensions from entities."""
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",
                "item_category": "fruit"
            },
            "tokyo": {
                "dimension": "Location",
                "specific_place": "tokyo",
                "place_type": "city"
            },
            "paris": {
                "dimension": "Location",
                "specific_place": "paris",
                "place_type": "city"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        assert len(derived) == 2
        assert "Product" in derived
        assert "Location" in derived
        assert len(derived["Location"].hierarchy) >= 1

    def test_infer_hierarchy_order_from_position(self):
        """Test hierarchy order is inferred from field position."""
        # LLM outputs fields in specific-to-general order
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",      # Position 0 - most specific
                "item_category": "fruit",       # Position 1
                "broader_category": "produce"   # Position 2 - most general
            },
            "banana": {
                "dimension": "Product",
                "specific_item": "banana",
                "item_category": "fruit",
                "broader_category": "produce"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        hierarchy = derived["Product"].hierarchy
        # specific_item should come before item_category which should come before broader_category
        assert hierarchy.index("specific_item") < hierarchy.index("item_category")
        assert hierarchy.index("item_category") < hierarchy.index("broader_category")

    def test_handle_inconsistent_hierarchy_fields(self):
        """Test handling entities with different hierarchy fields."""
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",
                "item_category": "fruit",
                "brand_name": "Granny Smith"  # Only in this entity
            },
            "banana": {
                "dimension": "Product",
                "specific_item": "banana",
                "item_category": "fruit"
                # No brand_name
            },
            "orange": {
                "dimension": "Product",
                "specific_item": "orange",
                "item_category": "fruit"
                # No brand_name
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # brand_name appears in <50% of entities, should be excluded or warned
        hierarchy = derived["Product"].hierarchy
        # brand_name should either be excluded or there should be a warning
        if "brand_name" not in hierarchy:
            assert "Product" in warnings
            assert any("brand_name" in str(w) for w in warnings["Product"])

    def test_empty_entities(self):
        """Test handling empty entities dict."""
        entities = {}

        derived, warnings = derive_dimensions_from_entities(entities)

        assert derived == {}
        assert warnings == {}

    def test_entity_missing_dimension_field(self):
        """Test entities without dimension field are skipped."""
        entities = {
            "apple": {
                "specific_item": "apple"
                # Missing 'dimension' field
            },
            "tokyo": {
                "dimension": "Location",
                "specific_place": "tokyo"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Only Location should be derived (apple skipped)
        assert len(derived) == 1
        assert "Location" in derived

    def test_merge_with_existing_dimensions(self):
        """Test merging derived dimensions with existing ones."""
        existing = {
            "Product": DiscoveredDimensionSchema(
                hierarchy=["specific_item", "item_category"],
                description="Products for sale",
                examples=["apple"]
            )
        }

        entities = {
            "banana": {
                "dimension": "Product",
                "specific_item": "banana",
                "item_category": "fruit",
                "origin_country": "Ecuador"  # New field
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities, existing)

        # Should extend hierarchy with new field
        assert "Product" in derived
        assert "origin_country" in derived["Product"].hierarchy
        # Should preserve existing description
        assert "Products for sale" in derived["Product"].description

    def test_generate_meaningful_description(self):
        """Test description generation includes examples."""
        entities = {
            "apple": {"dimension": "Product", "name": "apple"},
            "banana": {"dimension": "Product", "name": "banana"},
            "orange": {"dimension": "Product", "name": "orange"}
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        description = derived["Product"].description
        assert "Product" in description
        # Should include some example entities
        assert any(ex in description for ex in ["apple", "banana", "orange"])

    def test_hierarchy_defaults_to_dimension_name(self):
        """Test hierarchy defaults to dimension name if no fields."""
        entities = {
            "apple": {
                "dimension": "Product"
                # No hierarchy fields
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Should have at least one hierarchy level
        assert len(derived["Product"].hierarchy) >= 1
        # Default should be based on dimension name
        assert "product" in derived["Product"].hierarchy[0].lower()

    def test_case_sensitivity_in_dimension_names(self):
        """Test dimension names preserve case."""
        entities = {
            "apple": {"dimension": "Product", "name": "apple"},
            "banana": {"dimension": "product", "name": "banana"}  # lowercase
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Should treat "Product" and "product" as different dimensions
        # (case-sensitive by default, merging handled separately)
        assert len(derived) == 2
        assert "Product" in derived
        assert "product" in derived

    def test_non_dict_entity_data_skipped(self):
        """Test non-dict entity data is skipped gracefully."""
        entities = {
            "apple": {
                "dimension": "Product",
                "name": "apple"
            },
            "invalid": "not a dict",
            "also_invalid": 123
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Should only process the valid entity
        assert len(derived) == 1
        assert "Product" in derived


class TestDimensionDerivationIntegration:
    """Integration tests for dimension derivation with Pydantic models."""

    def test_derived_schema_is_valid_pydantic(self):
        """Test derived schema passes Pydantic validation."""
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",
                "item_category": "fruit"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Should be valid DiscoveredDimensionSchema
        schema = derived["Product"]
        assert isinstance(schema, DiscoveredDimensionSchema)
        # Hierarchy should be normalized (snake_case)
        for level in schema.hierarchy:
            assert level == level.lower()
            assert ' ' not in level

    def test_derived_schema_get_field_definitions(self):
        """Test derived schema can generate field definitions."""
        entities = {
            "apple": {
                "dimension": "Product",
                "specific_item": "apple",
                "item_category": "fruit"
            }
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        fields = derived["Product"].get_field_definitions()
        assert len(fields) >= 1
        assert all("name" in f and "type" in f for f in fields)

    def test_examples_limited_to_reasonable_count(self):
        """Test examples are limited to prevent bloat."""
        # Create many entities
        entities = {
            f"entity_{i}": {
                "dimension": "LargeDimension",
                "name": f"entity_{i}"
            }
            for i in range(100)
        }

        derived, warnings = derive_dimensions_from_entities(entities)

        # Examples should be limited (max 5 for new dimensions)
        assert len(derived["LargeDimension"].examples) <= 5

    def test_merged_examples_limited(self):
        """Test merged examples are limited to prevent bloat."""
        existing = {
            "Product": DiscoveredDimensionSchema(
                hierarchy=["name"],
                description="Products",
                examples=["a", "b", "c", "d", "e", "f", "g", "h"]  # 8 examples
            )
        }

        entities = {
            f"new_{i}": {"dimension": "Product", "name": f"new_{i}"}
            for i in range(10)
        }

        derived, warnings = derive_dimensions_from_entities(entities, existing)

        # Examples should be limited to 10
        assert len(derived["Product"].examples) <= 10
