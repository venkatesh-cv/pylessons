import logging

def add(x,y):
    logger = logging.getLogger("exampleApp.otherModule")
    logger.info("added %i and %i" % (x,y))
    return x+y