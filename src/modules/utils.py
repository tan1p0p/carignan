import logging

def show_logs():
    logging.basicConfig(
        format="%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
        level=logging.DEBUG
    )
