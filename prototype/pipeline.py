from dag import DAG

class Pipeline:
    
    def __init__(self):
        self.tasks = DAG()
        
    def task(self, depends_on=None):
        def inner(f):
            self.tasks.add(f)
            if depends_on:
                self.tasks.add(depends_on, f)
            return f
        return inner
    
    def run(self, param = None):
        order = self.tasks.sort()
        outputs = {}
        for n, task in enumerate(order):
            for k,v in self.tasks.graph.items():
                if task in v:
                    outputs[task] = task(outputs[k])
            if task not in outputs:
                if n == 0 and param:
                    outputs[task] = task(param)
                else:
                    outputs[task] = task()               
        return outputs