import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.environ.get('P2C_CODE', ROOT)          # lets the same tests run against another checkout
sys.path.insert(0, ROOT)
sys.path.insert(0, CODE)
