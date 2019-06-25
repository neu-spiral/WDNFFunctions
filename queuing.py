import random
from simpy import *
import logging

class Message:
    def __init__(self, payload):
        self.payload = [payload]
        self.counter = 1

class Queue:
    """M/M/1 queue."""
    def __init__(self, env, service_rate, output=None):
        self.env = env
        self.service_rate= service_rate
        self.input = Store(self.env)
        self.customers = 0  # record the number of customers in queue
        if output:
            self.output = output
        else:
            self.output = Store(self.env)
        self.env.process(self.serve())

    def enqueue(self, msg):
        self.input.put(msg)
        self.customers += 1

    def serve(self):
        while True:
            msg = yield self.input.get()
            yield self.env.timeout(random.expovariate(self.service_rate))
            logging.debug("pop a message" + str(msg.payload))
            self.customers -=1
            self.output.put(msg)

    def get_custormers(self):
        return self.customers

class MMInfQueue(Queue):
    """M/M/Infinity queue"""
    def serve(self):
        while True:
            msg = yield self.input.get()
            self.env.process(self.process_msg(msg))

    def process_msg(self,msg):
        yield self.env.timeout(random.expovariate(self.service_rate))
        logging.debug("pop a message " + str(msg.payload))
        self.customers -= 1
        self.output.put(msg)

class CountQueue(Queue):
    """
    A count queue. If a message is currently served, additional messages "attach themselves" to it, incrementing its counter.
    Messages should contain a counter field, a payload field.
    """

    def __init__(self, env, service_rate, output=None):
        Queue.__init__(self, env, service_rate, output)
        self.pending_messages = []

    def enqueue(self, msg):
        if self.customers == 0:
            self.input.put(msg)
        else:
            self.pending_messages.extend(msg.payload)
        self.customers += msg.counter

    def serve(self):
        while True:
            msg = yield self.input.get()
            self.env.process(self.process_msg(msg))

    def process_msg(self,msg):
        yield self.env.timeout(random.expovariate(self.service_rate))
        logging.debug("pop a message " + str(msg.payload) + " with " + str(self.customers))
        if self.customers > msg.counter:
            msg.counter = self.customers
            msg.payload.extend(self.pending_messages)
        elif self.customers < msg.counter:
            raise Exception("message merge wrong!")
        self.output.put(msg)
        self.customers = 0
        # test if msg.counter == len(msg.payload)
        if msg.counter != len(msg.payload):
            raise Exception("message merge wrong!")
        self.pending_messages = []


class MultiQueue(Queue):
    """queue with multiple service rates"""
    def __init__(self, env, QueueType, service_rates, output=None):
        self.service_rates = service_rates
        self.env = env
        if output:
            self.output = output
        else:
            self.output = Store(self.env)
        self.queues = {}
        for d in service_rates:
            self.queues[d] = QueueType(self.env, self.service_rates[d], self.output)

    def enqueue(self, msg, d):
        self.queues[d].enqueue(msg)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    def sender(queue, env, message, d=None):
        if d:
            while True:
                msg = Message(str(message)+ " at time "+ str(env.now))
                queue.enqueue(msg, d)
                logging.debug("push a meesage "+ str(d) + " :"+ str(message) + " at time "+str(env.now))
                yield env.timeout(4)
        else:
            while True:
                msg = Message(str(message)+ " at time "+ str(env.now))
                queue.enqueue(msg)
                logging.debug("push a meesage "+ str(message)+ " at time "+str(env.now))
                yield env.timeout(4)

    env = Environment()
    store = Store(env)
    service_rates = {1:0.1, 2:0.2, 3:0.3}
    message1 = 1
    message2 = 2
    message3 = 3



#    q1 = Queue(env, service_rates[1], store)
#    env.process(sender(q1, env, message1))

#    q2 = MultiQueue(env, MMInfQueue, service_rates, store)
#    env.process(sender(q2, env, message1,1))
#    env.process(sender(q2, env, message2,2))
#    env.process(sender(q2, env, message3,3))

#    q3 = MMInfQueue(env, service_rates[1], store)
#    env.process(sender(q3, env, message1))

    q4 = CountQueue(env, service_rates[1], store)
    env.process(sender(q4, env, message1))

    env.run(100)