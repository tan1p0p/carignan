class SampleClass():
    def __init__(self, x):
        self.x = x
    
    def call_func1(self):
        func1(self.x)

def func1(x):
    print(x)