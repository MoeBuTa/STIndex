"""
Tool definitions for agentic spatiotemporal extraction workflow.

The LLM can call these tools to normalize temporal expressions and geocode locations.
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
import json


class ToolParameter(BaseModel):
    """Tool parameter specification."""
    name: str
    type: str
    description: str
    required: bool = True


class Tool(BaseModel):
    """Tool specification for LLM tool calling."""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Registry of available tools for spatiotemporal extraction."""

    def __init__(
        self,
        time_normalizer=None,
        geocoder=None,
        enable_temporal: bool = True,
        enable_spatial: bool = True
    ):
        """
        Initialize ToolRegistry.

        Args:
            time_normalizer: EnhancedTimeNormalizer instance
            geocoder: EnhancedGeocoderService instance
            enable_temporal: Enable temporal tools
            enable_spatial: Enable spatial tools
        """
        self.time_normalizer = time_normalizer
        self.geocoder = geocoder
        self.enable_temporal = enable_temporal
        self.enable_spatial = enable_spatial

        # Register tools
        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available tools."""

        if self.enable_temporal and self.time_normalizer:
            # Temporal normalization tool
            self.tools["normalize_temporal"] = Tool(
                name="normalize_temporal",
                description=(
                    "Normalize a temporal expression to ISO 8601 format. "
                    "Use this for dates, times, durations, and intervals. "
                    "Examples: 'March 15, 2022' -> '2022-03-15', "
                    "'3 PM' -> '15:00:00', 'for 2 hours' -> 'PT2H'"
                ),
                parameters=[
                    ToolParameter(
                        name="temporal_expression",
                        type="string",
                        description="The temporal expression to normalize (e.g., 'March 15, 2022', 'yesterday', '3 PM')",
                        required=True
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Surrounding context to help with disambiguation",
                        required=False
                    ),
                    ToolParameter(
                        name="document_years",
                        type="array",
                        description="List of years mentioned in the document for year inference",
                        required=False
                    )
                ],
                function=self._normalize_temporal
            )

        if self.enable_spatial and self.geocoder:
            # Geocoding tool
            self.tools["geocode_location"] = Tool(
                name="geocode_location",
                description=(
                    "Get latitude/longitude coordinates for a location using authoritative geocoding API. "
                    "Use this instead of generating coordinates yourself. "
                    "Examples: 'Paris, France' -> (48.8566, 2.3522), "
                    "'Broome, Western Australia' -> (-17.9614, 122.2359)"
                ),
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="The location name to geocode (e.g., 'Paris, France', 'Broome, Western Australia')",
                        required=True
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Surrounding context for disambiguation",
                        required=False
                    ),
                    ToolParameter(
                        name="parent_region",
                        type="string",
                        description="Parent region for disambiguation (e.g., 'Western Australia' for 'Broome')",
                        required=False
                    )
                ],
                function=self._geocode_location
            )

            # Location disambiguation tool
            self.tools["disambiguate_location"] = Tool(
                name="disambiguate_location",
                description=(
                    "Get detailed information about a location including full name, country, and admin area. "
                    "Useful for ambiguous locations like 'Paris' (France? Texas?)"
                ),
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="The location name to disambiguate",
                        required=True
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Context to help disambiguation",
                        required=False
                    )
                ],
                function=self._disambiguate_location
            )

    def _normalize_temporal(
        self,
        temporal_expression: str,
        context: Optional[str] = None,
        document_years: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Normalize temporal expression using EnhancedTimeNormalizer.

        Args:
            temporal_expression: Temporal expression to normalize
            context: Context for disambiguation
            document_years: Years mentioned in document

        Returns:
            Normalized result
        """
        try:
            # Use batch normalization for context awareness
            normalized, temporal_type = self.time_normalizer.normalize_batch(
                [(temporal_expression, context or "")],
                document_text=None
            )[0]

            return {
                "success": True,
                "original": temporal_expression,
                "normalized": normalized,
                "type": temporal_type.value if temporal_type else "unknown",
                "iso_8601": normalized
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original": temporal_expression
            }

    def _geocode_location(
        self,
        location_name: str,
        context: Optional[str] = None,
        parent_region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Geocode location using EnhancedGeocoderService.

        Args:
            location_name: Location name
            context: Context for disambiguation
            parent_region: Parent region hint

        Returns:
            Geocoding result with coordinates
        """
        try:
            coords = self.geocoder.get_coordinates(
                location_name,
                context=context,
                parent_region=parent_region
            )

            if coords:
                lat, lon = coords
                return {
                    "success": True,
                    "location": location_name,
                    "latitude": lat,
                    "longitude": lon,
                    "source": "nominatim"
                }
            else:
                return {
                    "success": False,
                    "error": "Location not found",
                    "location": location_name
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "location": location_name
            }

    def _disambiguate_location(
        self,
        location_name: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed location information.

        Args:
            location_name: Location name
            context: Context for disambiguation

        Returns:
            Location details
        """
        try:
            details = self.geocoder.get_location_details(
                location_name,
                context=context
            )

            if details:
                return {
                    "success": True,
                    "location": location_name,
                    "full_name": details.get("display_name"),
                    "latitude": details.get("latitude"),
                    "longitude": details.get("longitude"),
                    "country": details.get("country"),
                    "admin_area": details.get("state"),
                    "locality": details.get("city")
                }
            else:
                return {
                    "success": False,
                    "error": "Location not found",
                    "location": location_name
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "location": location_name
            }

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }

        tool = self.tools[tool_name]
        if not tool.function:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' has no implementation"
            }

        try:
            return tool.function(**kwargs)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in LangChain/OpenAI format.

        Returns:
            List of tool definitions
        """
        definitions = []

        for tool in self.tools.values():
            properties = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                if param.required:
                    required.append(param.name)

            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })

        return definitions

    def get_tool_schemas_for_prompt(self) -> str:
        """
        Get tool schemas formatted for inclusion in prompts.

        Returns:
            Formatted tool schemas string
        """
        schemas = []

        for tool in self.tools.values():
            params = []
            for param in tool.parameters:
                required_str = " (required)" if param.required else " (optional)"
                params.append(f"  - {param.name} ({param.type}){required_str}: {param.description}")

            schema = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters:
{chr(10).join(params)}
"""
            schemas.append(schema)

        return "\n".join(schemas)

    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
