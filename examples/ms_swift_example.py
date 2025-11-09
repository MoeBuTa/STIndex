"""
MS-SWIFT Direct Integration Example.

Uses MS-SWIFT functions directly without wrapper classes:
- swift.llm.run_deploy() for server deployment
- swift.llm.inference_client() for inference
- DeployArguments and XRequestConfig for configuration
"""

from stindex.utils.config import load_config_from_file
from stindex.llm.ms_swift import MSSwiftLLM, deploy_server


def main():
    """Example: Deploy server and run extraction."""
    print("=" * 80)
    print("MS-SWIFT Direct Integration Example")
    print("=" * 80)

    # Load configuration
    config = load_config_from_file("ms_swift")

    # Deploy server using swift.llm.run_deploy()
    print("\nDeploying MS-SWIFT server (using swift.llm.run_deploy)...")
    with deploy_server(config) as port:
        print(f"✓ Server running on port {port}")

        # Create client (uses swift.llm.inference_client)
        llm_config = config.get("llm", {})
        client = MSSwiftLLM(llm_config)

        # Example text
        text = """
        On March 15, 2022, Tropical Cyclone Seroja made landfall near Kalbarri,
        Western Australia. The category 3 cyclone caused widespread damage across
        the Mid West region. By March 17, recovery efforts had begun in Geraldton.
        """

        # Extract spatiotemporal information
        messages = [
            {
                "role": "user",
                "content": f"Extract temporal and spatial entities from: {text}"
            }
        ]

        print("\nGenerating response...")
        result = client.generate(messages)

        if result.success:
            print(f"\n✓ Success!")
            print(f"Model: {result.model}")
            print(f"Client mode: {'swift.llm.inference_client' if client.use_swift else 'OpenAI SDK'}")
            print(f"\nResponse:\n{result.content}")
            print(f"\nTokens: {result.usage.total_tokens}")
        else:
            print(f"\n✗ Failed: {result.error_msg}")

    print("\n✓ Server stopped")


def example_with_stindex():
    """Example: Use with STIndexExtractor."""
    print("\n" * 2)
    print("=" * 80)
    print("Using MS-SWIFT with STIndexExtractor")
    print("=" * 80)

    from stindex import STIndexExtractor

    # Load config and deploy server
    config = load_config_from_file("ms_swift")

    print("\nDeploying server...")
    with deploy_server(config) as port:
        print(f"✓ Server on port {port}")

        # Use STIndexExtractor with ms_swift config
        extractor = STIndexExtractor(config_path="ms_swift")

        text = "On March 15, 2022 in Broome, Australia, a cyclone occurred."

        print(f"\nExtracting from: {text}")
        result = extractor.extract(text)

        if result.success:
            print(f"\n✓ Extraction successful!")
            print(f"\nTemporal entities: {len(result.temporal_entities)}")
            for entity in result.temporal_entities:
                print(f"  - {entity.text} → {entity.normalized}")

            print(f"\nSpatial entities: {len(result.spatial_entities)}")
            for entity in result.spatial_entities:
                print(f"  - {entity.text}")
                if entity.latitude and entity.longitude:
                    print(f"    ({entity.latitude}, {entity.longitude})")
        else:
            print(f"\n✗ Extraction failed: {result.error_message}")

    print("\n✓ Server stopped")


if __name__ == "__main__":
    # Run basic example
    main()

    # Uncomment to try with STIndexExtractor
    # example_with_stindex()
