
<img src="./img/Neurons-Network_T.jpg">

<p> <center>
    <font face='Helvetica' size='5.6'><b>Code Your Own RNN with NumPy</b></font>
    </center>
</p>
<img src="./img/Neurons-Network_B.jpg">

---

<h2>What is a Recurrent Neural Network and How Do They Work?</h2>

The neural networks we have seen throughout this Winter School treat training data as independent, isolated events. In other words, we don’t treat and/or make use of sequential data. Therefore, in order to process a temporal series of data points (e.g. seismograms) or a sequence of event (e.g. text) you would have to feed the entire sequence to the neural network at once!  

This doesn’t make sense both on a computation-level and a human-level.  Think about it, as you read text you are storing a subset of this text in your short-term memory; you are keeping the sequence of words that are relevant for your understanding of the sentence.  

This is the idea behind Recurrent Neural Networks.  A <i>recurrent neural network</i> (RNN) processes sequences by iterating through the sequence of elements and maintaining a <i>state</i> containing information relative to what it has seen so far. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being dependent on the previous computations.  

In other words, data points are no longer processed in a single step. The network will loop over itself until it performs the same task on each element on the sequence.  The RNN will reset itself only when it reaches the final element.   

Let's visualize this before going through an example. Below we see a typical RNN:  

![image.png](attachment:image.png)
<b>Left </b>A single recurrent network, which is nothing more than a network with a loop. <b>Right </b> The same RNN but <i>unrolled</i> for visualization purposes.

$x_{t}$ and $s_{t}$ are the input and hidden state (both vectors), respectively, at time $t$. Matrices $U$, $V$, and $W$ are the parameters we want to learn from our data. And $o_{t}$ is the output vector computed using only the hidden state at time $ = t$.  The hidden state, $s_{t}$, is calculated based on the previous hidden state ($s_{t-1}$) and the input at the current step, $x_{t}$: 

$$ s_{t} = f(U * x_{t} + W * s_{t-1}) $$  

i.e. $s_{t}$ kept information on what happened from all of the previous steps. It can be thought as a memory object. Note that, unlike other typical neural networks, recurrent neural networks reuse the same parameters (weights) $U$, $V$, and $W$ during the training process.  This makes sense since we are performing the same task on each element of the time sequence, $x_{t}$.

Our activation function, $f$, is defined either as an $tanh$ or $ReLU$. For example, when $f$ is defined as $tanh$, our hidden state becomes:  

$$ s_{t} = tanh(U * x_{t} + W * s_{t-1}) $$

<h2>Writting your own RNN using Numpy</h2>

Let's do some programming to get a deep (pun intended) understanding of recurrent neural networks. To do this we will create a generic and simple RNN using Numpy.  The objective of this exercise it to understand on a basic level how an RNN operates. We will not worry about using real data for now. Instead, let's create random data and feed this to an RNN. We will say that our input vector has $32$ input features (this is what goes into our input layer) and we will have an output vector with $64$ features. 


Below are the ingredients you'll need and some psuedo code to get you started. 

<b>RNN ingredients:</b>  

1] Define the dimension of the input space. [This it the input layer of our neural network]. 

2] Define the dimension of the output feature space [Output layer of our neural networt].  

3]  Generate random noise as our input ‘data’ [We just want to get an idea of HOW this works].  
&emsp; <b>Hint:</b> Input vector should have dimensions (timesteps X input_features)

4] Define an initial state, $s_{t}$, of the RNN.  

5] Create (random) weight matrices, $W$ and $U$.  

6] Create a for loop. that takes in the input with the current state (the previous output) to obtain the current output.  
&emsp;<b> Hint:</b> Don't forget about our activation function, $f$.  

7] Update the state for the next step.

8] The final output should be a 2D tensor with dimensions (timesteps, output_features).
### Pseudocode for RNN ###

# For the first timestep, the previous output isn’t defined; 
# Thus, our initial state is set to zero.
state_t = 0

# Iterates over sequnce elements
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    # The previous output becomes the state for the next iteration.
    state_t  = output_t
<h3>How this would look like using code:</h3>


```python
### Create fake data and store them as a tensor:
# Number of timesteps in the input sequence.
timesteps = 100

# Dimensionality of the input feature space
input_features = 32

# Dimensionality of the output feature space.
output_features = 64

# Input data is random for the sake of it
inputs = np.random.random((timesteps, input_features))


### RNN ###
# Initialize state: an all-zero vector
state_t = np.zeros((output_features))

# The RNN's parameters are two matrices W and U and a bias vector.
# Initialize random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    # Combines the input with the current state (the previous output)
    # to obtain the current output
    output_t = np.tanh( np.dot(W, input_t) + np.dot(U, state_t) + b )
    
    # Stores this output in a list
    successive_outputs.append(output_t)
    
    # Use for the next round:
    state_t = output_t

# The final output is a 2D tensor of shape (timesteps, output features)
final_output_sequence = np.concatenate(successive_outputs, axis = 0)
```

---

<h2>Building a DNN in Keras</h2>

Before jumping in to defining an RNN using Keras, let's remind ourselves what pieces we need to compile DNN using Keras' high-level neural network API. The ingredients are:  

<b> Define a model</b>. In TensorFlow there are two ways to do this: first, by using a <span style="font-family:Courier; font-size:1.1em;">Sequential</span> class or second, with a <span style="font-family:Courier; font-size:1.1em;">functional API</span> (which allows you to build arbritrary model structures).  Given that the most common neural network configuration is made up of linear stacks, you'll most likely use the first method more often.  

<b>Define your DNN by stacking layers</b> We start from  the input layer and <i>sequentially</i> add more
layers: the hidden layers, and the output layer.(Hence, where the <span style="font-family:Courier; font-size:1.1em;">Class</span> name came from).  Basically, the more layers you stack together, the more complex model you define (but at the potential expense of overfitting and/or long computation times!).  

One thing to keep in mind: each layer you define needs to be compatible with the next. In other words, each layer will only accept and return tensors of a certain shape.  

<b>Compile</b> Before the training occurs, you need to define compile the model. To do this, use Keras' [<span style="font-family:Courier; font-size:1.1em;">compile</span>](https://keras.io/getting-started/sequential-model-guide/#compilation) method.  This method takes in three arguments:  
1) the [optimizer](https://keras.io/optimizers/) - optimization algorithm (e.g. SGD, Adam, etc.); 2) the [loss](https://keras.io/losses/) function the optimizer will try to minimize; and 3) a list of metrics (e.g. accuracy).    

And that's it! The last step is to train the model using Keras' <span style="font-family:Courier; font-size:1.1em;">fit</span> function. More on this when we run our own RNN in the next notebook.

---

With the above knowledge fresh in our memory we could replace our Numpy RNN (lines 17 - 39) with this ONE line:  

<br><center><span style="font-family:Courier; font-size:1.1em;">model.add(SimpleRNN( ))</span>  </center>

For example:   


```python
from keras.models import Sequential
from keras import layers 
from keras.layers import Embedding, Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1)))
model.add(Dense(1))
```

    Using TensorFlow backend.


Let's look at this line by line:  

<b>Line 5:</b> Defined our model architect using a <span style="font-family:Courier; font-size:1.1em;">Sequential</span> class. 

<b>Line 6:</b> Added our RNN layer (which also serves as our input layer).

<b>Line 7:</b> Added a fully connected (i.e. Dense) layer as our output layer. 

The <span style="font-family:Courier; font-size:1.1em;">model.summary()</span> function is a convenient way of checking how our deep neural network textually looks like. It provides key information of our architecture such as:  

the <b>layer type</b> and the order of the layers from input (first row) to output (bottom row before the '=');  
the <b>shape of the tensor</b> for each output (and thus, what is going into the next layer);  
and the <b>number of weights</b> (here labeled 'parameters') per layer along with a summary of the total number of weights.  

For example:  


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn_1 (SimpleRNN)     (None, 10)                120       
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 131
    Trainable params: 131
    Non-trainable params: 0
    _________________________________________________________________


What do we see? The first line is our header  

<center>[ <span style="font-family:Courier; font-size:1.1em;">Layer(type)</span>, <span style="font-family:Courier; font-size:1.1em;"> Output Shape,</span>, and <span style="font-family:Courier; font-size:1.1em;"> Param # </span>]</center>   
Where <span style="font-family:Courier; font-size:1.1em;">Output Shape</span> is the shape of the tensor that is leaving our first layer (<span style="font-family:Courier; font-size:1.1em;">SimpleRNN</span>) and going into the next layer <span style="font-family:Courier; font-size:1.1em;">Dense</span> (i.e. a fully connected layer).     

In the next line We see that we have an output shape of (None, 10) and 120 Parameters:<br>  

<center><span style="font-family:Courier; font-size:1.1em;">simple_rnn_1 (SimpleRNN) (None, 10) 120 </span>  </center>  

What does this mean? When we wrote line 6:  
<center><span style="font-family:Courier; font-size:1.1em;">SimpleRNN(10, input_shape=(3, 1))</span></center>

We specified that we had 10 weights (parameters) and input shape of (3,1). The 3 here means we have 3 sequences(e.g. three timeseries points) we want to input and 1 featuere (e.g. Temperature).  

OK, now to the weights.  From the output above we have 120 parameters. Why do we have 120 parameters?  

Remember, there are two things going on with our simple RNN: First you have the recurrent loop, where the state is fed recurrently into the model to generate the next step.  Weights for the recurrent step are:    

<center><b>recurrent_weights</b> = num_units * num_units</center>

Second, there is a new input of your sequence at each step:
<br><center><b>input_weights</b> = num_features * num_units</center>

So now we have the weights, whats missing are the biases - for every unit one bias:

<center><b>biases</b> = num_units * 1</center>  

In our case we have that num_units = $10$ and num_features = $1$.  

Putting this altogether we have the following formula for the number of parameters/weights:  
<br><center><span style="font-family:Courier; font-size:1.1em;">Parameters</span> = num_units x num_units + num_features x num_units + biases</center>

Where <b>num_units</b> is the number of weights in the RNN (<span style="font-family:Courier; font-size:1.1em;">10</span>) and <b>num_features</b> is the number features of our input. (In thie case <span style="font-family:Courier; font-size:1.1em;">1</span>).  
<br><center><span style="font-family:Courier; font-size:1.1em;">Parameters</span> = $10 * 10 + 1 * 10 + 10 = 120$</center>  

Finally, we have our output layer. In this example we defined it as a <span style="font-family:Courier; font-size:1.1em;">Dense</span> layer:   
<br>
<center><span style="font-family:Courier; font-size:1.1em;">Dense(1)</span>  </center>  

So this last <span style="font-family:Courier; font-size:1.1em;">Dense</span> layer takes its input (<span style="font-family:Courier; font-size:1.1em;">10</span> (the output of the previous layer) and adds the bias to give us <span style="font-family:Courier; font-size:1.1em;">11</span> parameters/weights. Since we defined the dense layer as: <span style="font-family:Courier; font-size:1.1em;">Dense(1)</span> we are telling our neural network that we want a single output.



---

Great! So you now how they work and you just looked through a short example of how you could implement a RNN model using Keras.  In the next notebook we will do a full runthrough of creating and running a RNN on real data.  

Just as an aside, while the RNN we defined was cute and simple, in practice these simple RNNs are not used. The reason? Well, we run into the problem of vanishing gradients and exploding gradients.  The vanishing gradient problem is why folks use the more exotic recurrent neural network known as long short term memory (LSTM).  I won't go over the details here, but you can think of LSTMs as an extended version of recurrent neural networks. LSTMs act like computers in that they can read, delete, and add to their 'stored' memory. This allows them to 'extend' their 'short term memory' to 'longer short term memory'.

If you are curious, I suggest checking this blog by <a href ='https://medium.com/deep-math-machine-learning-ai/chapter-10-1-deepnlp-lstm-long-short-term-memory-networks-with-math-21477f8e4235'>Madhu Sanjeevi</a>. It's one of my favorite explainations of LSTMs.

---
---

<h3>About this Notebook</h3>  

The above example and code is from Ch.6 of Chollet's <i>Deep Learning with Python</i>[1]. Content was added for further clarification and readability.

Jupyter Notebook by: Noah Luna    

<h3>Suggested Reading Material</h3>  

Géron, A. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow:  
&emsp;Concepts, Tools, and Techniques for Building Intelligent Systems. O'Reilly UK Ltd.  

Karpath, Andrej. "The Unreasonable Effectiveness of Recurrent Neural Networks"  
&emsp; <i>Andrej Karpathy blog</i>, 24 March 2017, http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 

Sanjeevi, Madhu. "DeepNLP — LSTM (Long Short Term Memory) Networks with Math" <i>Medium</i>, 21 Jan 2018,  
&emsp; https://medium.com/deep-math-machine-learning-ai/chapter-10-1-deepnlp-lstm-long-short-term-memory-networks-with-math-21477f8e4235  


<h3>Sources</h3>  

[1]Chollet, F. (2017). Deep Learning with Python. Manning Publications.  

<h3>Jupyter Notebook by:</h3>  
Noah Grayson Luna
