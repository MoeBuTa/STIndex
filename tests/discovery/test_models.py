"""
Unit tests for schema discovery Pydantic models.

Tests validation, field validators, helper methods, and serialization.
"""

import pytest
from pydantic import ValidationError

from stindex.discovery.models import (
    DiscoveredDimensionSchema,
    HierarchicalEntity,
    ClusterSchemaDiscoveryResult,
    DimensionSource,
    MergedDimensionSchema,
    FinalSchema,
    merge_dimension_schemas
)


class TestDiscoveredDimensionSchema:
    """Tests for DiscoveredDimensionSchema model."""

    def test_valid_schema(self):
        """Test creating valid schema."""
        schema = DiscoveredDimensionSchema(
            hierarchy=['specific_symptom', 'symptom_category'],
            description='Medical symptoms',
            examples=['fever', 'cough']
        )
        assert schema.hierarchy == ['specific_symptom', 'symptom_category']
        assert schema.description == 'Medical symptoms'
        assert schema.examples == ['fever', 'cough']

    def test_hierarchy_normalization(self):
        """Test hierarchy levels are normalized to snake_case."""
        schema = DiscoveredDimensionSchema(
            hierarchy=['Specific Symptom', 'SYMPTOM CATEGORY'],
            description='Test'
        )
        assert schema.hierarchy == ['specific_symptom', 'symptom_category']

    def test_empty_hierarchy_fails(self):
        """Test empty hierarchy raises validation error."""
        with pytest.raises(ValidationError):
            DiscoveredDimensionSchema(
                hierarchy=[],
                description='Test'
            )

    def test_get_field_definitions(self):
        """Test converting hierarchy to field definitions."""
        schema = DiscoveredDimensionSchema(
            hierarchy=['specific_drug', 'drug_class'],
            description='Pharmacology'
        )
        fields = schema.get_field_definitions()
        assert len(fields) == 2
        assert fields[0] == {
            'name': 'specific_drug',
            'type': 'string',
            'description': 'Specific Drug level'
        }
        assert fields[1] == {
            'name': 'drug_class',
            'type': 'string',
            'description': 'Drug Class level'
        }


class TestHierarchicalEntity:
    """Tests for HierarchicalEntity model."""

    def test_valid_entity(self):
        """Test creating valid entity."""
        entity = HierarchicalEntity(
            text='fever',
            dimension='symptom',
            hierarchy_values={
                'specific_symptom': 'fever',
                'symptom_category': 'systemic'
            },
            confidence=0.95
        )
        assert entity.text == 'fever'
        assert entity.dimension == 'symptom'
        assert entity.confidence == 0.95

    def test_text_validation_strips_whitespace(self):
        """Test entity text is stripped."""
        entity = HierarchicalEntity(
            text='  fever  ',
            dimension='symptom'
        )
        assert entity.text == 'fever'

    def test_empty_text_fails(self):
        """Test empty text raises validation error."""
        with pytest.raises(ValidationError):
            HierarchicalEntity(
                text='',
                dimension='symptom'
            )

    def test_whitespace_only_text_fails(self):
        """Test whitespace-only text raises validation error."""
        with pytest.raises(ValidationError):
            HierarchicalEntity(
                text='   ',
                dimension='symptom'
            )

    def test_confidence_bounds(self):
        """Test confidence must be in [0, 1]."""
        # Valid confidence
        entity = HierarchicalEntity(text='fever', dimension='symptom', confidence=0.5)
        assert entity.confidence == 0.5

        # Invalid confidence (< 0)
        with pytest.raises(ValidationError):
            HierarchicalEntity(text='fever', dimension='symptom', confidence=-0.1)

        # Invalid confidence (> 1)
        with pytest.raises(ValidationError):
            HierarchicalEntity(text='fever', dimension='symptom', confidence=1.1)

    def test_get_hierarchy_value(self):
        """Test getting value at specific hierarchy level."""
        entity = HierarchicalEntity(
            text='fever',
            dimension='symptom',
            hierarchy_values={
                'specific_symptom': 'fever',
                'symptom_category': 'systemic'
            }
        )
        assert entity.get_hierarchy_value('specific_symptom') == 'fever'
        assert entity.get_hierarchy_value('symptom_category') == 'systemic'
        assert entity.get_hierarchy_value('nonexistent') is None

    def test_matches_entity_identical(self):
        """Test matching identical entities."""
        entity1 = HierarchicalEntity(text='fever', dimension='symptom')
        entity2 = HierarchicalEntity(text='fever', dimension='symptom')
        assert entity1.matches_entity(entity2, similarity_threshold=0.85)

    def test_matches_entity_case_insensitive(self):
        """Test matching is case-insensitive."""
        entity1 = HierarchicalEntity(text='Fever', dimension='symptom')
        entity2 = HierarchicalEntity(text='fever', dimension='symptom')
        assert entity1.matches_entity(entity2, similarity_threshold=0.85)

    def test_matches_entity_similar(self):
        """Test matching similar entities."""
        entity1 = HierarchicalEntity(text='influenza', dimension='disease')
        entity2 = HierarchicalEntity(text='influenzas', dimension='disease')
        # Very similar (singular vs plural) - should match with 0.85 threshold
        assert entity1.matches_entity(entity2, similarity_threshold=0.85)

    def test_matches_entity_different(self):
        """Test non-matching different entities."""
        entity1 = HierarchicalEntity(text='fever', dimension='symptom')
        entity2 = HierarchicalEntity(text='cough', dimension='symptom')
        assert not entity1.matches_entity(entity2, similarity_threshold=0.85)


class TestClusterSchemaDiscoveryResult:
    """Tests for ClusterSchemaDiscoveryResult model."""

    def test_valid_result(self):
        """Test creating valid cluster result."""
        result = ClusterSchemaDiscoveryResult(
            cluster_id=0,
            n_questions=100,
            discovered_dimensions={
                'symptom': DiscoveredDimensionSchema(
                    hierarchy=['specific_symptom', 'symptom_category'],
                    description='Medical symptoms'
                )
            },
            entities={
                'symptom': [
                    HierarchicalEntity(text='fever', dimension='symptom'),
                    HierarchicalEntity(text='cough', dimension='symptom')
                ]
            }
        )
        assert result.cluster_id == 0
        assert result.n_questions == 100
        assert len(result.discovered_dimensions) == 1
        assert len(result.entities['symptom']) == 2

    def test_get_entity_count(self):
        """Test getting entity count for dimension."""
        result = ClusterSchemaDiscoveryResult(
            cluster_id=0,
            n_questions=100,
            entities={
                'symptom': [
                    HierarchicalEntity(text='fever', dimension='symptom'),
                    HierarchicalEntity(text='cough', dimension='symptom')
                ],
                'disease': [
                    HierarchicalEntity(text='pneumonia', dimension='disease')
                ]
            }
        )
        assert result.get_entity_count('symptom') == 2
        assert result.get_entity_count('disease') == 1
        assert result.get_entity_count('nonexistent') == 0

    def test_get_total_entities(self):
        """Test getting total entities across all dimensions."""
        result = ClusterSchemaDiscoveryResult(
            cluster_id=0,
            n_questions=100,
            entities={
                'symptom': [
                    HierarchicalEntity(text='fever', dimension='symptom'),
                    HierarchicalEntity(text='cough', dimension='symptom')
                ],
                'disease': [
                    HierarchicalEntity(text='pneumonia', dimension='disease')
                ]
            }
        )
        assert result.get_total_entities() == 3

    def test_get_dimension_stats(self):
        """Test getting entity counts per dimension."""
        result = ClusterSchemaDiscoveryResult(
            cluster_id=0,
            n_questions=100,
            entities={
                'symptom': [
                    HierarchicalEntity(text='fever', dimension='symptom'),
                    HierarchicalEntity(text='cough', dimension='symptom')
                ],
                'disease': [
                    HierarchicalEntity(text='pneumonia', dimension='disease')
                ]
            }
        )
        stats = result.get_dimension_stats()
        assert stats == {'symptom': 2, 'disease': 1}


class TestDimensionSource:
    """Tests for DimensionSource model."""

    def test_add_cluster(self):
        """Test adding cluster sources."""
        source = DimensionSource()
        source.add_cluster(0, 10)
        source.add_cluster(1, 5)

        assert source.cluster_ids == {0, 1}
        assert source.entity_counts_per_cluster == {0: 10, 1: 5}

    def test_get_total_contributing_clusters(self):
        """Test getting number of contributing clusters."""
        source = DimensionSource()
        source.add_cluster(0, 10)
        source.add_cluster(1, 5)
        source.add_cluster(2, 3)

        assert source.get_total_contributing_clusters() == 3


class TestMergedDimensionSchema:
    """Tests for MergedDimensionSchema model."""

    def test_add_entities_from_cluster(self):
        """Test adding entities from a cluster."""
        schema = MergedDimensionSchema(
            hierarchy=['specific_symptom', 'symptom_category'],
            description='Medical symptoms'
        )
        entities = [
            HierarchicalEntity(text='fever', dimension='symptom'),
            HierarchicalEntity(text='cough', dimension='symptom')
        ]
        schema.add_entities_from_cluster(0, entities)

        assert len(schema.entities) == 2
        assert schema.total_entity_count == 2
        assert 0 in schema.sources.cluster_ids

    def test_merge_schema_from_cluster(self):
        """Test merging schema information from cluster."""
        schema = MergedDimensionSchema(
            hierarchy=['specific_symptom', 'symptom_category'],
            description='Medical symptoms',
            examples=['fever']
        )
        cluster_schema = DiscoveredDimensionSchema(
            hierarchy=['specific_symptom', 'symptom_category'],
            description='Medical symptoms',
            examples=['cough', 'headache']
        )
        schema.merge_schema_from_cluster(cluster_schema, 0)

        assert 'cough' in schema.examples
        assert 'headache' in schema.examples
        assert 0 in schema.sources.cluster_ids


class TestFinalSchema:
    """Tests for FinalSchema model."""

    def test_get_dimension_names(self):
        """Test getting sorted dimension names."""
        schema = FinalSchema(
            dimensions={
                'symptom': MergedDimensionSchema(
                    hierarchy=['specific_symptom'],
                    description='Symptoms'
                ),
                'disease': MergedDimensionSchema(
                    hierarchy=['specific_disease'],
                    description='Diseases'
                )
            },
            n_clusters_processed=5,
            total_questions_processed=500
        )
        names = schema.get_dimension_names()
        assert names == ['disease', 'symptom']  # Sorted

    def test_get_total_entities(self):
        """Test getting total entities across all dimensions."""
        schema = FinalSchema(
            dimensions={
                'symptom': MergedDimensionSchema(
                    hierarchy=['specific_symptom'],
                    description='Symptoms',
                    total_entity_count=10
                ),
                'disease': MergedDimensionSchema(
                    hierarchy=['specific_disease'],
                    description='Diseases',
                    total_entity_count=5
                )
            },
            n_clusters_processed=5,
            total_questions_processed=500
        )
        assert schema.get_total_entities() == 15

    def test_get_dimension_entity_counts(self):
        """Test getting entity counts per dimension."""
        schema = FinalSchema(
            dimensions={
                'symptom': MergedDimensionSchema(
                    hierarchy=['specific_symptom'],
                    description='Symptoms',
                    total_entity_count=10
                ),
                'disease': MergedDimensionSchema(
                    hierarchy=['specific_disease'],
                    description='Diseases',
                    total_entity_count=5
                )
            },
            n_clusters_processed=5,
            total_questions_processed=500
        )
        counts = schema.get_dimension_entity_counts()
        assert counts == {'symptom': 10, 'disease': 5}

    def test_to_yaml_dict(self):
        """Test converting to YAML-friendly dict."""
        entity1 = HierarchicalEntity(
            text='fever',
            dimension='symptom',
            hierarchy_values={'specific_symptom': 'fever', 'symptom_category': 'systemic'}
        )
        schema = FinalSchema(
            dimensions={
                'symptom': MergedDimensionSchema(
                    hierarchy=['specific_symptom', 'symptom_category'],
                    description='Medical symptoms',
                    examples=['fever'],
                    entities=[entity1],
                    total_entity_count=1,
                    sources=DimensionSource(cluster_ids={0})
                )
            },
            n_clusters_processed=1,
            total_questions_processed=100
        )
        yaml_dict = schema.to_yaml_dict()

        assert 'symptom' in yaml_dict
        assert yaml_dict['symptom']['hierarchy'] == ['specific_symptom', 'symptom_category']
        assert yaml_dict['symptom']['count'] == 1
        assert len(yaml_dict['symptom']['entities']) == 1
        assert yaml_dict['symptom']['entities'][0]['text'] == 'fever'


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_merge_dimension_schemas_single(self):
        """Test merging single schema returns itself."""
        schema = DiscoveredDimensionSchema(
            hierarchy=['specific_symptom'],
            description='Symptoms'
        )
        merged = merge_dimension_schemas([schema], 'symptom')
        assert merged.hierarchy == ['specific_symptom']
        assert merged.description == 'Symptoms'

    def test_merge_dimension_schemas_multiple(self):
        """Test merging multiple schemas."""
        schema1 = DiscoveredDimensionSchema(
            hierarchy=['specific_symptom'],
            description='Medical symptoms',
            examples=['fever']
        )
        schema2 = DiscoveredDimensionSchema(
            hierarchy=['specific_symptom', 'symptom_category'],
            description='Patient symptoms',
            examples=['cough']
        )
        merged = merge_dimension_schemas([schema1, schema2], 'symptom')

        # Should use longest hierarchy
        assert merged.hierarchy == ['specific_symptom', 'symptom_category']
        # Should combine descriptions
        assert 'Medical symptoms' in merged.description
        assert 'Patient symptoms' in merged.description
        # Should merge examples
        assert 'fever' in merged.examples
        assert 'cough' in merged.examples

    def test_merge_dimension_schemas_empty_fails(self):
        """Test merging empty list raises error."""
        with pytest.raises(ValueError):
            merge_dimension_schemas([], 'symptom')
