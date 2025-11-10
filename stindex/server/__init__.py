"""Server module for STIndex.

Note: vLLM server and router have been replaced by MS-SWIFT.
For server deployment, use MS-SWIFT instead:

    from stindex.llm.ms_swift import deploy_server

    config = load_config_from_file("ms_swift")
    with deploy_server(config) as port:
        # Server running on port
        ...
"""
