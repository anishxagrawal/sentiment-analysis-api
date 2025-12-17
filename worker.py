import os
from redis import Redis
from rq import Worker, Queue, Connection

# Connect to Redis
redis_conn = Redis(host='localhost', port=6379, db=0)

if __name__ == '__main__':
    with Connection(redis_conn):
        worker = Worker(['default'])
        worker.work()