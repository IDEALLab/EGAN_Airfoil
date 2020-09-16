#%%
class A:
    def __init__(self):
        super().__init__()
        self.a = 10
        print('sb')

class B(A):
    def b(self):
        pass

x = B()
x.__init__()
