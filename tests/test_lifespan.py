import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from fastapi.testclient import TestClient
from nava_core.gathi.api.main import app

def test_lifespan():
    with TestClient(app) as client:
        print("Retriever initialized:", hasattr(app.state, "rag_retriever"))

if __name__ == "__main__":
    test_lifespan()
