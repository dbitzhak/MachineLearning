import numpy as np 
import matplotlib.pyplot as plt
import random
import sys
import _pickle as pickle

"""
General Information:
    We are using a 2 layer Neural Network with Stochastic Gradient Descent

    X - MNIST fashion 28x28 pixels = 785 features with values between 0-255

    Y - 10 categories:
        0. T-shirt/top 
        1. Trouser
        2. Pullover
        3. Dress
        4. Coat
        5. Sandal
        6. Shirt
        7. Sneaker
        8. Bag
        9. Ankle boot

    Hyperparameters:
        Batch size
        learning rate
        number of units per hidden  layer
        activation functions
        epochs

    Activation functions:
        ReLU for hidden layer
        TanhH for hidden layer
        Softmax function for the output

Vectorization of batch operations:

    Fashion MNIST - each row an example, each column a pixel value

    number of features - F
    number of examples - E (batch size)
    K - nodes in layer i

    X*W = 

    W-layer i           X matrix       
    |            |      |   -x0-     | 
    |  |     |   |      |            | 
    |  wi0  wij  |      |            | 
    |  |     |   |      |   -xe-     | 
    |            |      |            |  
    |            | FxK  |            |ExF        


    = Z                 Bi matrix  - biases matrix     
    |    -zi0-   |      |    -b0-    | 
    |            |      |            | 
    |    -zij-   |  +   |    -be-    | 
    |            |      |            | 
    |            |      |            |  
    |            | ExK  |            |ExK

    activation of (ExK) = ExK input for next layer
"""

class NeuralNetwork:
    
    """
     Args:
        layer_list (list of ints): A list where layer_list[0] = number of units in hidden layer 0. The input layer and output layer not included.
        act_func (function): Activation function for hidden layers
        d_act_func (function): Derivative of the activation function for the hidden layers   
    """
    def __init__(self, hidden_layers,act_func,d_act_func):
       self.layers = hidden_layers #List of layers and their respective number of units
       self.activation = act_func  #Non linear acitvation function
       self.d_activation = d_act_func #Derivative of the activation

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x, numerically stable way."""
        """ Meaning: avoid very large exponents by reducing the largest to zero
            and everything else to less than that"""
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex, axis = 1, keepdims=True) #Sums over rows np.sum([[0, 1], [0, 5]], axis=1) = array([1, 5])

    def d_neg_log_likelihood(self,y_hat,y):
        num_examp = y.shape[0] #Number of examples
        gradient = np.copy(y_hat) #For safe manipulation
        gradient[range(num_examp),y.astype(int)] -= 1
        gradient = gradient/num_examp
        return gradient

    def train(self,train_X, train_Y, learning_rate, epochs,batch_size, num_classes):
        #Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dim = num_classes
        #Initializes values and run training
        self.set_model(train_X,train_Y)
        self.run_training(train_X, train_Y,)
        return self

    def set_model(self,X,Y):
        self.input_dim = X.shape[1] # Get number of columns - number of input features
        self.params = dict()
        self.initialize_weights() #Initializes weight and bias arrays to distinct random values

    def initialize_weights(self):
        #Initializes all weights and biases to a random value
        i = 1
        self.params["W0"] = np.random.uniform(-0.07,0.07,(self.input_dim, self.layers[0]))
        self.params["b0"] =  np.random.uniform(-0.07,0.07,(1,self.layers[0]))
        while i < len(self.layers):
            self.params["W"+str(i)] =np.random.uniform(-0.07,0.07,(self.layers[i-1], self.layers[i]))
            self.params["b"+str(i)] = np.random.uniform(-0.07,0.07,(1,self.layers[i]))
            i += 1
        self.params["W"+str(i)] = np.random.uniform(-0.07,0.07,(self.layers[i-1],self.output_dim))
        self.params["b"+str(i)] =  np.random.uniform(-0.07,0.07,(1,self.output_dim))

    def run_training(self,X,Y):
        for epoch in range(self.epochs):
            sum_loss = 0.0
            shuffle_pair(X,Y)
            for batch in range(0, X.shape[1], self.batch_size):
                x,y = X[batch:(batch+batch_size),:], Y[batch:(batch+batch_size)]
                output = self.forward(x)
                sum_loss = self.loss(output,y)
                self.backpropagate(x,y)
            if(epoch%10 ==0):
                print epoch
                print sum_loss
            
    def backpropagate(self,x,y):
        derivatives = dict() #Store the derivatives
        index = len(self.layers) #Number of hidden layers
        #Output layer
        dloss = self.d_neg_log_likelihood(self.params["a"+str(index)],y) #dL/dzi
        #Hidden Layers
        while(index > 0):
            derivatives["W"+str(index)] = self.params["a" + str(index-1)].T.dot(dloss) #  dL/dzi * dzi/dwi  
            derivatives["b"+str(index)] = np.sum(dloss,axis=0,keepdims=True) #  dL/dzi * dzi/dbi
            dloss = dloss.dot(self.params["W"+str(index)].T)*(
                self.d_activation(self.params["z"+str(index-1)])) #  dL/dzi * dzi/dh(i-1) * dh(i-1)/dz(i-1)
            index -= 1 #Previous layer
        derivatives["W0"] = x.T.dot(dloss)
        derivatives["b0"] = np.sum(dloss,axis=0,keepdims=True)
        #Update step
        for key,value in derivatives.items():
            self.params[key] -= self.learning_rate*value

    def forward(self,X):
        i = 1
        #For the first layer (whose input is X)
        self.params["z0"] = X.dot(self.params["W0"]) + self.params["b0"] #X*W+b
        self.params["a0"] = self.activation(self.params["z0"]) #Non linear activation function
        # For the hidden layers (in case we wish to have more than one hidden layer)
        while i < len(self.layers):
            self.params["z"+str(i)] = self.params["a"+str(i-1)].dot(self.params["W"+ str(i)]) + self.params["b"+ str(i)] #ai-1*Wi+bi
            self.params["a"+str(i)] = self.activation(self.params["z"+str(i)]) #Non-linear activation function
            i += 1
        #Output layer
        self.params["z"+str(i)] = self.params["a"+str(i-1)].dot(self.params["W"+str(i)]) + self.params["b"+str(i)] #ai*W_out+b_out
        self.params["a"+str(i)] = self.softmax(self.params["z"+str(i)]) #Softmax instead of activation
        return np.copy(self.params["a"+str(i)]) #Allows for safe manipulation
         
    def predict(self,X):
        results = self.forward(X) #Runs the forward pass
        return np.argmax(results, axis=1) #Returns the index of the max value in each row 
            
    def loss(self,output,y):
        output = np.copy(output) #Copy array for safe manipulation
        log_loss = -np.log(output[range(self.batch_size), y.astype(int)]) #Log of the output
        log_loss = np.sum(log_loss) #Total loss for all the checked examples
        return np.true_divide(log_loss,batch_size) #Averaged loss

#Possible activation functions and their derivations
def softmax(x):
    """Compute softmax values for each sets of scores in x, numerically stable way."""
    """ Meaning: avoid very large exponents by reducing the largest to zero
    and everything else to less than that, so they go to zero instead of infinity an Nan"""
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=1, keepdims = True)

def relu(x):
    #If(x>=0) -> 1*x = x else 0*x=0
    return(x * (x >= 0))

def d_relu(x):
    return (x > 0) * 1 #If(x>=0) = 1 else 0 (in theory it's undefined for zero)

def d_tanh(x):
   return (1 - np.power((np.tanh(x)),2)) # 1 - sqrt(tanh)

def shuffle_pair(x,y):
        """ Zips and shuffles both items as a pair, unziping afterwards"""
        xy = list(zip(x,y))
        np.random.shuffle(xy)
        x, y = zip(*xy)
        x = np.array(x)
        y = np.array(y)

def load_data():
    temp = open('train_x.pkl', 'rb')
    train_x = pickle.load(temp)
    dev_size = int(train_x.shape[0]*0.2)
    train_x = np.true_divide(train_x, 255) #normalization
    temp = open('train_y.pkl', 'rb')
    train_y = pickle.load(temp)
    temp = open('test_x.pkl', 'rb')
    test_x = pickle.load(temp)
    test_x = np.true_divide(test_x, 255) #normalization
    shuffle_pair(train_x,train_y)
    dev_x, dev_y = train_x[-dev_size:,:],train_y[-dev_size:]
    train_x,train_y = train_x[:-dev_size,:],train_y[:-dev_size]
    return train_x,train_y,dev_x,dev_y,test_x

#For pickling the data
"""train_x=np.loadtxt('train_x')
pickle.dump(train_x,open('train_x.pkl','wb'))
train_y=np.loadtxt('train_y')
pickle.dump(train_y,open('train_y.pkl','wb'))
test_x=np.loadtxt('test_x')
pickle.dump(test_x,open('test_x.pkl','wb'))"""



train_x,train_y,dev_x,dev_y,test_x = load_data()
#Hyperparameters
hidden_layer_size = int((2/3)*train_x.shape[1]+10) #Number of classes 
hidden_layer = [hidden_layer_size]
learning_rate = 0.1
epochs = 200
batch_size = 150
nn = NeuralNetwork(hidden_layer,relu,d_relu)
model = nn.train(train_x, train_y, learning_rate,epochs,batch_size,10)
print "Trained!"
errors = 0
for x,y in zip (dev_x,dev_y):
    ans = model.predict(x)
    if (ans != y):
        errors += 1
error_percentage = errors/dev_x.shape[0]
print error_percentage
with open('test.pred', 'w') as pred_file:
    for x in test_x:
        ans = str(nn.predict(x)[0])
        pred_file.write(ans+"\n")

