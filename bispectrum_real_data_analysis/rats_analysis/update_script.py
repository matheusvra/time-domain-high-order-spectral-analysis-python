import os
import shutil
from loguru import logger

script_to_be_updated = ''

logger.info(f"Updating {script_to_be_updated} in all folders.")

cwd = '/'.join(__file__.split('/')[:-1])

logger.info(f"Current working directory: {cwd}")

folders = list(os.listdir(cwd))

makefile_path = os.path.join(cwd, f'Template/{script_to_be_updated}')


for folder in folders:
    
    if folder.startswith('group'):
        logger.info(f"Updating {script_to_be_updated} in {folder}.")
        shutil.copy2(makefile_path, os.path.join(cwd, folder))

logger.success(f"Updated {script_to_be_updated} in all folders.")