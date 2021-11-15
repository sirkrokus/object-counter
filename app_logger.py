import logging
import logging.handlers


def init():
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    logger = logging  # .getLogger()
    # logging.basicConfig(format='%(asctime)d-%(levelname)s-%(message)s')

    # Create formatters
    c_format = logging.Formatter('%(levelname)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(c_format)

    # file_handler = logging.FileHandler('app.log')
    file_handler = logging.handlers.TimedRotatingFileHandler('app.log', when='midnight', backupCount=10)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(f_format)

    # Add handlers to the logger
    # logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger
