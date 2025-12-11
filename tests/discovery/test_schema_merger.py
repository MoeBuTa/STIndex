"""
Unit tests for SchemaMerger.

Tests dimension alignment, entity deduplication, and schema merging.
"""

import pytest
from stindex.discovery.schema_merger import SchemaMerger
from stindex.discovery.models import (
    ClusterSchemaDiscoveryResult,
    DiscoveredDimensionSchema,
    HierarchicalEntity,
    FinalSchema
)


class TestSchemaMerger:
    """Tests for SchemaMerger class."""

    def test_initialization(self):
        """Test SchemaMerger initialization."""
        merger = SchemaMerger(similarity_threshold=0.85)
        assert merger.similarity_threshold == 0.85

    def test_are_similar_identical(self):
        """Test similarity check with identical strings."""
        merger = SchemaMerger(similarity_threshold=0.85)
        assert merger._are_similar('symptom', 'symptom')

    def test_are_similar_case_insensitive(self):
        """Test similarity check is case-insensitive."""
        merger = SchemaMerger(similarity_threshold=0.85)
        assert merger._are_similar('Symptom', 'symptom')

    def test_are_similar_high_similarity(self):
        """Test similarity check with high similarity."""
        merger = SchemaMerger(similarity_threshold=0.85)
        # 'disease' vs 'diseases' are very similar
        assert merger._are_similar('disease', 'diseases')

    def test_are_similar_low_similarity(self):
        """Test similarity check with low similarity."""
        merger = SchemaMerger(similarity_threshold=0.85)
        # 'symptom' vs 'disease' are not similar
        assert not merger._are_similar('symptom', 'disease')

    def test_align_dimensions_identical_names(self):
        """Test dimension alignment with identical names across clusters."""
        merger = SchemaMerger(similarity_threshold=0.85)

        # Create cluster results with identical dimension names
        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Medical symptoms'
                    )
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Patient symptoms'
                    )
                }
            )
        ]

        groups = merger._align_dimensions(cluster_results)

        # Should group both under 'symptom'
        assert len(groups) == 1
        assert 'symptom' in groups
        assert len(groups['symptom']) == 2

    def test_align_dimensions_similar_names(self):
        """Test dimension alignment with similar but not identical names."""
        merger = SchemaMerger(similarity_threshold=0.85)

        # Create cluster results with similar dimension names
        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'disease': DiscoveredDimensionSchema(
                        hierarchy=['specific_disease'],
                        description='Diseases'
                    )
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'diseases': DiscoveredDimensionSchema(
                        hierarchy=['specific_disease'],
                        description='Diseases'
                    )
                }
            )
        ]

        groups = merger._align_dimensions(cluster_results)

        # Should group both under canonical name (first occurrence)
        assert len(groups) == 1
        assert 'disease' in groups or 'diseases' in groups
        canonical_name = list(groups.keys())[0]
        assert len(groups[canonical_name]) == 2

    def test_align_dimensions_different_names(self):
        """Test dimension alignment with completely different names."""
        merger = SchemaMerger(similarity_threshold=0.85)

        # Create cluster results with different dimension names
        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Symptoms'
                    )
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'disease': DiscoveredDimensionSchema(
                        hierarchy=['specific_disease'],
                        description='Diseases'
                    )
                }
            )
        ]

        groups = merger._align_dimensions(cluster_results)

        # Should create separate groups
        assert len(groups) == 2
        assert 'symptom' in groups
        assert 'disease' in groups

    def test_deduplicate_hierarchical_entities_identical(self):
        """Test deduplication removes identical entities."""
        merger = SchemaMerger(similarity_threshold=0.85)

        entities = [
            HierarchicalEntity(text='fever', dimension='symptom'),
            HierarchicalEntity(text='fever', dimension='symptom'),
            HierarchicalEntity(text='cough', dimension='symptom')
        ]

        unique = merger._deduplicate_hierarchical_entities(entities)

        # Should keep only 2 unique entities
        assert len(unique) == 2
        entity_texts = [e.text for e in unique]
        assert 'fever' in entity_texts
        assert 'cough' in entity_texts

    def test_deduplicate_hierarchical_entities_case_insensitive(self):
        """Test deduplication is case-insensitive."""
        merger = SchemaMerger(similarity_threshold=0.85)

        entities = [
            HierarchicalEntity(text='Fever', dimension='symptom'),
            HierarchicalEntity(text='fever', dimension='symptom'),
            HierarchicalEntity(text='FEVER', dimension='symptom')
        ]

        unique = merger._deduplicate_hierarchical_entities(entities)

        # Should keep only 1 entity (all variants of 'fever')
        assert len(unique) == 1
        assert unique[0].text in ['Fever', 'fever', 'FEVER']

    def test_deduplicate_hierarchical_entities_similar(self):
        """Test deduplication handles similar entities."""
        merger = SchemaMerger(similarity_threshold=0.85)

        entities = [
            HierarchicalEntity(text='influenza', dimension='disease'),
            HierarchicalEntity(text='influenzas', dimension='disease'),
            HierarchicalEntity(text='pneumonia', dimension='disease')
        ]

        unique = merger._deduplicate_hierarchical_entities(entities)

        # Should keep 2 entities (influenza/influenzas are similar)
        assert len(unique) == 2
        entity_texts = [e.text for e in unique]
        # Either 'influenza' or 'influenzas' should be kept, plus 'pneumonia'
        assert 'pneumonia' in entity_texts

    def test_merge_clusters_single_cluster(self):
        """Test merging single cluster result."""
        merger = SchemaMerger(similarity_threshold=0.85)

        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=100,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom', 'symptom_category'],
                        description='Medical symptoms',
                        examples=['fever', 'cough']
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(
                            text='fever',
                            dimension='symptom',
                            hierarchy_values={'specific_symptom': 'fever', 'symptom_category': 'systemic'}
                        ),
                        HierarchicalEntity(
                            text='cough',
                            dimension='symptom',
                            hierarchy_values={'specific_symptom': 'cough', 'symptom_category': 'respiratory'}
                        )
                    ]
                }
            )
        ]

        final_schema = merger.merge_clusters(cluster_results)

        assert isinstance(final_schema, FinalSchema)
        assert len(final_schema.dimensions) == 1
        assert 'symptom' in final_schema.dimensions
        assert final_schema.dimensions['symptom'].total_entity_count == 2
        assert final_schema.n_clusters_processed == 1
        assert final_schema.total_questions_processed == 100

    def test_merge_clusters_multiple_clusters_same_dimension(self):
        """Test merging multiple clusters with same dimension."""
        merger = SchemaMerger(similarity_threshold=0.85)

        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Symptoms'
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(text='fever', dimension='symptom'),
                        HierarchicalEntity(text='cough', dimension='symptom')
                    ]
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Symptoms'
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(text='fever', dimension='symptom'),  # Duplicate
                        HierarchicalEntity(text='headache', dimension='symptom')
                    ]
                }
            )
        ]

        final_schema = merger.merge_clusters(cluster_results)

        assert len(final_schema.dimensions) == 1
        assert 'symptom' in final_schema.dimensions
        # Should have 3 unique entities (fever, cough, headache) after deduplication
        assert final_schema.dimensions['symptom'].total_entity_count == 3
        assert final_schema.n_clusters_processed == 2
        assert final_schema.total_questions_processed == 100

    def test_merge_clusters_multiple_dimensions(self):
        """Test merging clusters with different dimensions."""
        merger = SchemaMerger(similarity_threshold=0.85)

        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Symptoms'
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(text='fever', dimension='symptom')
                    ]
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'disease': DiscoveredDimensionSchema(
                        hierarchy=['specific_disease'],
                        description='Diseases'
                    )
                },
                entities={
                    'disease': [
                        HierarchicalEntity(text='pneumonia', dimension='disease')
                    ]
                }
            )
        ]

        final_schema = merger.merge_clusters(cluster_results)

        assert len(final_schema.dimensions) == 2
        assert 'symptom' in final_schema.dimensions
        assert 'disease' in final_schema.dimensions
        assert final_schema.dimensions['symptom'].total_entity_count == 1
        assert final_schema.dimensions['disease'].total_entity_count == 1

    def test_merge_dimension_group(self):
        """Test merging dimension group from multiple clusters."""
        merger = SchemaMerger(similarity_threshold=0.85)

        # Create cluster results
        cluster_results = [
            ClusterSchemaDiscoveryResult(
                cluster_id=0,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom'],
                        description='Medical symptoms',
                        examples=['fever']
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(text='fever', dimension='symptom')
                    ]
                }
            ),
            ClusterSchemaDiscoveryResult(
                cluster_id=1,
                n_questions=50,
                discovered_dimensions={
                    'symptom': DiscoveredDimensionSchema(
                        hierarchy=['specific_symptom', 'symptom_category'],
                        description='Patient symptoms',
                        examples=['cough']
                    )
                },
                entities={
                    'symptom': [
                        HierarchicalEntity(text='cough', dimension='symptom')
                    ]
                }
            )
        ]

        # Create cluster_schemas list
        cluster_schemas = [
            (0, cluster_results[0].discovered_dimensions['symptom']),
            (1, cluster_results[1].discovered_dimensions['symptom'])
        ]

        merged_dim = merger._merge_dimension_group('symptom', cluster_schemas, cluster_results)

        # Should use longest hierarchy
        assert merged_dim.hierarchy == ['specific_symptom', 'symptom_category']
        # Should combine descriptions
        assert 'Medical symptoms' in merged_dim.description or 'Patient symptoms' in merged_dim.description
        # Should combine examples
        assert 'fever' in merged_dim.examples
        assert 'cough' in merged_dim.examples
        # Should have 2 unique entities
        assert merged_dim.total_entity_count == 2
        # Should track both cluster sources
        assert len(merged_dim.sources.cluster_ids) == 2
