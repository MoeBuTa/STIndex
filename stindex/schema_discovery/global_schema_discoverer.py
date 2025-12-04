"""
Global schema discoverer for cross-cluster dimensional discovery.

Discovers dimensional schemas from representative samples across ALL clusters
to ensure consistent dimensions used by all clusters.
"""

from typing import Dict, List
import json
from loguru import logger

from stindex.llm.manager import LLMManager
from stindex.llm.prompts.initial_schema_discovery import GlobalSchemaPrompt
from stindex.schema_discovery.cot_logger import CoTLogger


class GlobalSchemaDiscoverer:
    """
    Discover global dimensions from representative samples across all clusters.

    Simple approach: NO retrieval, NO embeddings - just LLM-based schema discovery.
    """

    def __init__(self, llm_config: Dict, output_dir: str = None):
        """
        Initialize global schema discoverer.

        Args:
            llm_config: LLM configuration dict (provider, model, etc.)
            output_dir: Output directory for CoT logging (optional)
        """
        self.llm_manager = LLMManager(llm_config)
        self.cot_logger = CoTLogger(output_dir) if output_dir else None

    def discover_dimensions(
        self,
        cluster_samples: Dict[int, List[str]]
    ) -> Dict[str, Dict]:
        """
        Discover global dimensions from representative samples across all clusters.

        Args:
            cluster_samples: Dict mapping cluster_id -> list of sample questions
                Example: {0: [20 questions], 1: [20 questions], ...}

        Returns:
            Dict mapping dimension name -> schema definition:
            {
                'symptom': {
                    'hierarchy': ['specific_symptom', 'symptom_category'],
                    'description': 'Patient symptoms and clinical signs',
                    'examples': ['fever', 'cough', 'headache']
                },
                'disease': {
                    'hierarchy': ['disease_name', 'disease_category', 'body_system'],
                    'description': 'Medical conditions and diseases',
                    'examples': ['pneumonia', 'diabetes', 'hypertension']
                },
                ...
            }
        """
        logger.info(f"Discovering global dimensions from {len(cluster_samples)} clusters")

        # Flatten all sample questions from all clusters
        all_samples = []
        for cluster_id in sorted(cluster_samples.keys()):
            samples = cluster_samples[cluster_id]
            all_samples.extend(samples)
            logger.debug(f"  Cluster {cluster_id}: {len(samples)} samples")

        total_samples = len(all_samples)
        logger.info(f"Total samples for global discovery: {total_samples}")

        # Create prompt for schema discovery
        # Discover ~5 dimensions (configurable based on domain)
        prompt = GlobalSchemaPrompt(
            n_schemas=5,  # Discover ~5 domain-specific dimensions
            predefined_dimensions=[],  # No predefined dimensions for simplicity
            cluster_id=None  # Global discovery across all clusters
        )

        # Build messages
        messages = prompt.build_messages(all_samples)

        logger.info("Calling LLM for global schema discovery...")
        logger.debug(f"  System prompt length: {len(messages[0]['content'])} chars")
        logger.debug(f"  User prompt length: {len(messages[1]['content'])} chars")

        # Get LLM response
        try:
            response = self.llm_manager.generate(messages)
            logger.info("✓ LLM response received")
            logger.debug(f"  Response length: {len(response.content)} chars")

        except Exception as e:
            logger.error(f"✗ LLM generation failed: {e}")
            raise

        # Parse discovered dimensions
        try:
            result = prompt.parse_response(response.content)
            discovered_dims = result['schemas']
            reasoning = result['reasoning']

            # Log CoT
            if self.cot_logger:
                self.cot_logger.log_global_discovery(
                    reasoning=reasoning,
                    raw_response=result['raw_response']
                )

            logger.info(f"✓ Parsed {len(discovered_dims)} dimensions")

            # Log discovered dimensions
            for dim_name, dim_schema in discovered_dims.items():
                hierarchy_str = ' → '.join(dim_schema.get('hierarchy', []))
                logger.info(f"  • {dim_name}: {hierarchy_str}")
                logger.debug(f"    Description: {dim_schema.get('description', 'N/A')}")

            return discovered_dims

        except Exception as e:
            logger.error(f"✗ Failed to parse LLM response: {e}")
            logger.debug(f"  Raw response: {response.content[:500]}")
            raise

    def discover_dimensions_from_file(
        self,
        cluster_samples_file: str
    ) -> Dict[str, Dict]:
        """
        Convenience method to discover dimensions from cluster samples JSON file.

        Args:
            cluster_samples_file: Path to cluster_samples.json
                Expected format: {0: [questions], 1: [questions], ...}

        Returns:
            Discovered dimensional schemas
        """
        logger.info(f"Loading cluster samples from: {cluster_samples_file}")

        try:
            with open(cluster_samples_file, 'r') as f:
                cluster_samples = json.load(f)

            # Convert string keys to integers if needed
            cluster_samples = {int(k): v for k, v in cluster_samples.items()}

            logger.info(f"Loaded samples from {len(cluster_samples)} clusters")

            return self.discover_dimensions(cluster_samples)

        except FileNotFoundError:
            logger.error(f"✗ Cluster samples file not found: {cluster_samples_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON in cluster samples file: {e}")
            raise
