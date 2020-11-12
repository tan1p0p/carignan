import test_class

print('=== A ===')
sample_class = test_class.SampleClass(1)
sample_class.call_func1()

def new_func(x):
    print(x + 100)

print('=== B ===')
test_class.func1 = new_func
sample_class.call_func1()