import logging
import sys

def init_logger(name:str, filename:str, level:int=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s|%(filename)s:%(lineno)d|%(levelname)s|%(message)s')
    formatter.datefmt = '%d-%m-%Y %H:%M:%S'

    # Handler
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    fileHandler = logging.FileHandler(filename=filename)
    fileHandler.setFormatter(formatter)

    # Add Handler
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    # This will prevent the root log to output the log to console
    logger.propagate = False