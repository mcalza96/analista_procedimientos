try:
    import langchain_classic
    print(f"langchain_classic found: {langchain_classic.__file__}")
    from langchain_classic.retrievers import EnsembleRetriever
    print("Found EnsembleRetriever in langchain_classic.retrievers")
except ImportError as e:
    print(f"ImportError: {e}")

