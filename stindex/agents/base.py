"""
Base agent implementing the observe-reason-act pattern.

Inspired by MetacogRAG and ReAct (Yao et al., 2023).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from loguru import logger


class BaseAgent(ABC):
    """
    Base class for agents implementing the observe-reason-act pattern.

    The pattern follows:
    1. OBSERVE: Process environment and extract relevant information
    2. REASON: Use LLM to reason about observations
    3. ACT: Execute actions based on reasoning (e.g., call tools, format output)
    """

    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """
        Initialize the base agent.

        Args:
            config: Configuration dictionary containing LLM settings and agent parameters
            llm: Optional pre-initialized LLM instance to share across agents (saves memory)
        """
        self.config = config
        self.llm = llm if llm is not None else self._initialize_llm(config)
        self.state = {}  # Internal state for the agent

    def _initialize_llm(self, config: Dict[str, Any]) -> Any:
        """
        Initialize the LLM based on configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Initialized LLM instance
        """
        llm_provider = config.get("llm_provider", "local").lower()

        try:
            if llm_provider == "local":
                from stindex.agents.llm.local import LocalLLM
                return LocalLLM(config)

            elif llm_provider == "openai":
                from stindex.agents.llm.api import OpenAILLM
                return OpenAILLM(config)

            elif llm_provider == "anthropic":
                from stindex.agents.llm.api import AnthropicLLM
                return AnthropicLLM(config)

            else:
                raise ValueError(
                    f"Unsupported LLM provider: {llm_provider}. "
                    f"Supported: 'local', 'openai', 'anthropic'"
                )

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> Any:
        """
        OBSERVE: Extract relevant information from environment.

        Args:
            environment: Dictionary containing environment state/data

        Returns:
            Agent-specific observation model (Pydantic BaseModel)
        """
        pass

    @abstractmethod
    def reason(self, observations: Any) -> Any:
        """
        REASON: Use LLM to reason about observations.

        Args:
            observations: Agent-specific observation model

        Returns:
            Agent-specific reasoning model with LLM output (Pydantic BaseModel)
        """
        pass

    @abstractmethod
    def act(self, reasoning: Any) -> Any:
        """
        ACT: Execute actions based on reasoning.

        This may involve:
        - Calling external tools/APIs
        - Formatting structured output
        - Updating agent state

        Args:
            reasoning: Agent-specific reasoning model

        Returns:
            Agent-specific action response model (Pydantic BaseModel)
        """
        pass

    def run(self, environment: Dict[str, Any]) -> Any:
        """
        Execute the full observe-reason-act cycle.

        This method orchestrates the workflow and is shared by all agents.

        Args:
            environment: Dictionary containing environment state/data

        Returns:
            Agent-specific action response

        Raises:
            Exception: If any step fails
        """
        try:
            # Step 1: Observe
            logger.debug("Agent observe phase started")
            observations = self.observe(environment)
            logger.debug(f"Observations: {observations}")

            # Step 2: Reason
            logger.debug("Agent reason phase started")
            reasoning = self.reason(observations)
            logger.debug(f"Reasoning: {reasoning}")

            # Step 3: Act
            logger.debug("Agent act phase started")
            action_response = self.act(reasoning)
            logger.debug(f"Action response: {action_response}")

            return action_response

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            raise

    def update_state(self, key: str, value: Any) -> None:
        """
        Update agent state.

        Args:
            key: State key
            value: State value
        """
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get agent state.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value
        """
        return self.state.get(key, default)

    def reset_state(self) -> None:
        """Reset agent state."""
        self.state = {}
