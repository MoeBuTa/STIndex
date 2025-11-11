"""
Categorical validation post-processor.

Validates that LLM-extracted categories match predefined allowed values
from dimension configuration, with fuzzy matching and normalization.
"""

from typing import Any, Dict, List, Optional, Set
from loguru import logger


class CategoricalValidator:
    """
    Validates categorical extractions against allowed values.

    Workflow:
    1. Extract allowed values from dimension config
    2. Normalize extracted category (lowercase, strip)
    3. Validate against allowed values with fuzzy matching
    4. Filter or flag invalid categories
    """

    def __init__(
        self,
        strict_mode: bool = False,
        case_sensitive: bool = False,
        allow_fuzzy_match: bool = True,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize categorical validator.

        Args:
            strict_mode: If True, reject entities with invalid categories (default: False)
                        If False, keep entities but set category to "unknown" and log warning
            case_sensitive: If True, require exact case match (default: False)
            allow_fuzzy_match: If True, allow fuzzy matching for misspellings (default: True)
            fuzzy_threshold: Minimum similarity for fuzzy match (0.0-1.0, default: 0.8)
        """
        self.strict_mode = strict_mode
        self.case_sensitive = case_sensitive
        self.allow_fuzzy_match = allow_fuzzy_match
        self.fuzzy_threshold = fuzzy_threshold

    def validate_entities(
        self,
        entities: List[Dict[str, Any]],
        dimension_config: Any,
        dimension_name: str
    ) -> List[Dict[str, Any]]:
        """
        Validate categorical entities against allowed values.

        Args:
            entities: List of extracted categorical entities
            dimension_config: DimensionConfig with field definitions
            dimension_name: Name of the dimension (e.g., "event", "entity")

        Returns:
            List of validated entities (filtered if strict_mode=True)
        """
        # Extract allowed values from dimension config
        allowed_values = self._extract_allowed_values(dimension_config)

        if not allowed_values:
            logger.warning(
                f"No allowed values defined for categorical dimension '{dimension_name}'. "
                f"Skipping validation."
            )
            return entities

        logger.debug(
            f"Validating {len(entities)} {dimension_name} entities against "
            f"{len(allowed_values)} allowed categories: {allowed_values}"
        )

        validated_entities = []
        invalid_count = 0
        corrected_count = 0

        for entity in entities:
            category = entity.get("category", "")

            # Validate and normalize category
            is_valid, normalized_category = self._validate_category(
                category, allowed_values, dimension_name
            )

            if is_valid:
                if normalized_category != category:
                    # Category was corrected via fuzzy matching or normalization
                    entity["category"] = normalized_category
                    entity["category_confidence"] = entity.get("category_confidence", 1.0) * 0.9
                    corrected_count += 1
                    logger.debug(
                        f"  Corrected category: '{category}' → '{normalized_category}' "
                        f"(confidence reduced to {entity['category_confidence']:.2f})"
                    )
                validated_entities.append(entity)
            else:
                invalid_count += 1
                if self.strict_mode:
                    # Reject entity in strict mode
                    logger.warning(
                        f"  ✗ Rejected entity with invalid category '{category}' "
                        f"(text: '{entity.get('text', '')}'). Allowed: {allowed_values}"
                    )
                else:
                    # Keep entity but set category to "unknown"
                    entity["category"] = "unknown"
                    entity["category_confidence"] = 0.0
                    logger.warning(
                        f"  ⚠ Invalid category '{category}' → 'unknown' "
                        f"(text: '{entity.get('text', '')}'). Allowed: {allowed_values}"
                    )
                    validated_entities.append(entity)

        if invalid_count > 0:
            logger.warning(
                f"Found {invalid_count} invalid categories for '{dimension_name}'. "
                f"{'Rejected' if self.strict_mode else 'Set to unknown'}."
            )
        if corrected_count > 0:
            logger.info(
                f"Corrected {corrected_count} categories via normalization/fuzzy matching"
            )

        logger.info(
            f"✓ Validated {dimension_name}: {len(validated_entities)}/{len(entities)} entities kept"
        )

        return validated_entities

    def _extract_allowed_values(self, dimension_config: Any) -> Set[str]:
        """
        Extract allowed values from dimension config fields.

        Looks for field with type="enum" and extracts its values.

        Args:
            dimension_config: DimensionConfig object

        Returns:
            Set of allowed category values (normalized if case_insensitive)
        """
        allowed_values = set()

        # Iterate through field definitions
        for field in dimension_config.fields:
            if field.get("type") == "enum" and field.get("name") == "category":
                values = field.get("values", [])
                if self.case_sensitive:
                    allowed_values = set(values)
                else:
                    allowed_values = {v.lower().strip() for v in values}
                break

        return allowed_values

    def _validate_category(
        self,
        category: str,
        allowed_values: Set[str],
        dimension_name: str
    ) -> tuple[bool, str]:
        """
        Validate and normalize a single category value.

        Args:
            category: Extracted category value
            allowed_values: Set of allowed category values
            dimension_name: Name of dimension (for logging)

        Returns:
            Tuple of (is_valid, normalized_category)
        """
        if not category:
            return False, ""

        # Normalize
        normalized = category if self.case_sensitive else category.lower().strip()

        # Exact match after normalization
        if normalized in allowed_values:
            return True, normalized

        # Fuzzy matching (if enabled)
        if self.allow_fuzzy_match:
            best_match = self._find_fuzzy_match(normalized, allowed_values)
            if best_match:
                logger.debug(f"  Fuzzy matched '{category}' → '{best_match}'")
                return True, best_match

        return False, ""

    def _find_fuzzy_match(
        self,
        category: str,
        allowed_values: Set[str]
    ) -> Optional[str]:
        """
        Find best fuzzy match from allowed values.

        Uses simple string similarity (Levenshtein-like).

        Args:
            category: Normalized category to match
            allowed_values: Set of allowed values

        Returns:
            Best matching allowed value, or None if no good match
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return None

        best_match = None
        best_ratio = 0.0

        for allowed_value in allowed_values:
            ratio = SequenceMatcher(None, category, allowed_value).ratio()
            if ratio > best_ratio and ratio >= self.fuzzy_threshold:
                best_ratio = ratio
                best_match = allowed_value

        return best_match
