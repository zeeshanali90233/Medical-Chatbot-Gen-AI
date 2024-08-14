import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

list_of_files=[
    "src/__init__.py", 
    "src/helper.py", 
    "src/prompt.py", 
    ".env", 
    "setup.py", 
    "research/trials.ipynb", 
    "app.py", 
    "store_index.py", 
    "static/", 
    "templates/chat.html", 
]

for filepath in list_of_files:
    path=Path(filepath)
    if not path.exists():
        logging.info(f"Creating {path}")
        if path.is_dir() or filepath.endswith("/"):
            path.mkdir(parents=True,exist_ok=True)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with open(path,"x") as f:
                pass
    else:
        logging.info(f"Already Exists {path}")
        