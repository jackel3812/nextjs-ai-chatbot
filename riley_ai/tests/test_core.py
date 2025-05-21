# test_core.py for Riley-Ai unit tests

def test_process_message():
    from riley_ai.jarvis.core import process_message
    assert process_message("Hello") == "Processed: Hello"
