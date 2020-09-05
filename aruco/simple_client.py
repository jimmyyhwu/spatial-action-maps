import pprint
from multiprocessing.connection import Client

address = ('localhost', 6000)
conn = Client(address, authkey=b'secret password')
pp = pprint.PrettyPrinter(indent=4)
while True:
    data = conn.recv()
    pp.pprint(data)
