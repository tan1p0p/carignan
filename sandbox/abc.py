from abc import ABCMeta, abstractmethod

# 抽象クラス
class Animal(metaclass=ABCMeta):
    @abstractmethod
    def sound(self):
        pass

# 抽象クラスを継承
class Cat(Animal):
    def sound(self):
        print("Meow")

if __name__ == "__main__":
    assert issubclass(Cat().__class__, Animal)
    assert isinstance(Cat(), Animal)