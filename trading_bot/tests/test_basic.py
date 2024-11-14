"""
Basic test file to verify pytest configuration
"""

def test_addition():
    """Simple test of addition"""
    assert 1 + 1 == 2

def test_string():
    """Simple test of string operation"""
    assert "hello" + " world" == "hello world"

def test_boolean():
    """Simple test of boolean operation"""
    assert True is True

if __name__ == "__main__":
    print("This test file contains basic tests")
