import nn

neural_network = nn.MLP(3, [4,4,1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

for k in range(100000):
    ypred = [neural_network(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in neural_network.parameters():
        p.grad = 0.0

    loss.backward()

    for p in neural_network.parameters():
        p.data += -0.1 * p.grad
        
    print(k, loss.data)
    
print('training-completed')
    


