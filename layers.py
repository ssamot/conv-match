from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
import theano.tensor as T
import theano
from keras.layers.core import  initializations
from keras.layers.core import activations
from keras.layers import Layer, Merge
from keras.layers.pooling import _Pooling1D
from keras.layers.recurrent import Recurrent
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import SimpleRNN
import numpy as np


class AttentionMerge(Merge):

    def __init__(self,final, match_mode,   *args, **kwargs):
        self.final = final
        self.match_mode = match_mode
        super(AttentionMerge, self).__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        # can only merge two layers
        memory = self.layers[0].output

        if(self.match_mode == "abs"):
            memory = memory.dimshuffle(1, 0, 2)



            diff = memory
            for i in range(1, len(self.layers)):
                question = self.layers[i].output
                #if(len(question.shape) < 3):
                question = question.dimshuffle(0, "x", 1)
                repeated = T.repeat(question, diff.shape[0], axis=1).dimshuffle(1, 0, 2)
                diff -= repeated
            #if(self.final):
            #
            tbr = abs(diff)*(memory)
            tbr = tbr.dimshuffle(1,0,2)
            return tbr
        if(self.mode == "concat"):
            memory = memory.dimshuffle(1, 0, 2)
            question = self.layers[1].output
            question = question.dimshuffle(0, "x", 1)
            diff = T.repeat(question, memory.shape[0], axis=1).dimshuffle(1, 0, 2)
            tbr = T.concatenate([diff, memory],axis=2)


            tbr = tbr.dimshuffle(1,0,2)
            return tbr
        elif(self.match_mode == "dot"):
            #diff = memory
            question = self.layers[1].output
            question = question.dimshuffle(0, "x", 1)
            #repeated = T.repeat(question, diff.shape[0], axis=1).dimshuffle(1, 0, 2)
            questions = question

            for i in range(2, len(self.layers)):
                question = self.layers[i].output
                #if(len(question.shape) < 3):
                question = question.dimshuffle(0, "x", 1)
                #repeated = T.repeat(question, diff.shape[0], axis=1).dimshuffle(1, 0, 2)
                questions +=question
            #if(self.final):
            tbr = K.batch_dot(memory, questions.dimshuffle(0,2,1))

            tbr = tbr *  memory

        #
        return tbr


    def _arguments_validation(self, layers, mode, concat_axis, dot_axes,
                              node_indices, tensor_indices):
       pass

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list  # must have multiple input shape tuples
        # case: callable self._output_shape
        if hasattr(self.mode, '__call__'):
            if hasattr(self._output_shape, '__call__'):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0][0],) + tuple(self._output_shape)
            else:
                # TODO: consider shape auto-inference with TF
                raise Exception('The Merge layer ' + self.name +
                                ' has a callable `mode` argument, ' +
                                'and we cannot infer its output shape because ' +
                                'no `output_shape` argument was provided.' +
                                'Make sure to pass a shape tuple (or a callable) ' +
                                '`output_shape` to Merge.')
        # pre-defined merge modes
        input_shapes = input_shape
        if self.mode in ['sum', 'mul', 'ave', 'max']:
            # all tuples in input_shapes should be the same
            return input_shapes[0]
        elif self.mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[-1:]:
                if output_shape[self.concat_axis] is None or shape[self.concat_axis] is None:
                    output_shape[self.concat_axis] = None
                    break
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.mode in ['dot', 'cos']:
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            shape1.pop(self.dot_axes[0])
            shape2.pop(self.dot_axes[1])
            shape2.pop(0)
            output_shape = shape1 + shape2
            if len(output_shape) == 1:
                output_shape += [1]
            return tuple(output_shape)


class ResettingRNN(SimpleRNN):
    def __init__(self,break_point,  *args, **kwargs):
        self.final = break_point

        super(ResettingRNN, self).__init__(*args, **kwargs)

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]
        reset_prev_output = states[3]


        prev_output = K.switch(theano.tensor.eq((self.counter.get_value())%self.final, 0),reset_prev_output,  prev_output)
        self.counter.set_value(self.counter.get_value() + 1)

        if self.consume_less == 'cpu':
            raise Exception("Not supported")
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))


        return output, [output]

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        #constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)
        constants = self.get_constants(x) + self.get_initial_states(x)
        print len(constants),"constants"

        #print len(constants), "initial_state"
        #exit()
        #counter = [K.variable(1)]
        self.counter = K.variable(0)
        initial_states#+=counter



        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output






# class AttentionRecurrent(Recurrent):
#     def __init__(self, output_dim, activation = "tanh", truncate_gradient=-1,  inner_init='orthogonal',**kwargs):
#         self.output_dim = output_dim
#
#
#         self.inner_init = initializations.get(inner_init)
#         self.inner_activation = activations.get(activation)
#         self.truncate_gradient = truncate_gradient
#
#         super(AttentionRecurrent, self).__init__(**kwargs)
#
#
#     def build(self, input_shape):
#         input_dim = input_shape[2]
#
#         self.W = self.inner_init((input_dim, 2))
#         self.b = shared_zeros((2))
#
#
#         self.trainable_weights = [
#             self.W, self.b
#         ]
#
#
#
#
#
#     def _step(self,
#               x_mem, x_att,
#               h_tm1
#               ):
#         z0 = x_att[:, 0].dimshuffle(0, "x")
#         z1 = x_att[:, 1].dimshuffle(0, "x")
#
#         h_t = z0 * h_tm1 + z1 * x_mem
#         return h_t
#
#     def call(self, x, mask=None):
#         X = x
#
#         x_mem = X.dimshuffle((1, 0, 2))
#
#         x_att = self.inner_activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)
#
#         outputs, updates = theano.scan(
#             self._step,
#             sequences=[x_mem, x_att],
#             outputs_info=[
#                 T.unbroadcast(alloc_zeros_matrix(X.shape[0], self.output_dim), 1)
#             ],
#             truncate_gradient=self.truncate_gradient,
#             go_backwards=False)
#
#
#
#
#         return outputs[-1]


from keras.layers.recurrent import time_distributed_dense

class AttentionRecurrent(SimpleRNN):


    def get_initial_states(self, x):
        output_dim = 2
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def preprocess_input(self, x):
        output_dim = 2
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, output_dim,
                                          timesteps)
        else:
            return x


    def build(self, input_shape):
        output_dim = 2
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((output_dim, output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((output_dim,), name='{}_b'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True


    def att_step(self,
              x_mem, x_att,
              h_tm1
              ):
        z0 = x_att[:, 0].dimshuffle(0, "x")
        z1 = x_att[:, 1].dimshuffle(0, "x")

        h_t = z0 * h_tm1 + z1 * x_mem
        return h_t


    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, x_att, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        X = x
        x_mem = X.dimshuffle((1, 0, 2))
        x_att = x_att.dimshuffle((1, 0, 2))



        outputs, updates = theano.scan(
            self.att_step,
            sequences=[x_mem, x_att],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[0], self.output_dim), 1)
            ],
            #truncate_gradient=self.truncate_gradient,
            go_backwards=False)

        return outputs[-1]








class AttentionPooling(_Pooling1D):



    def call(self, x, mask=None):
        output = x[:,  ::self.pool_size[0],:]
        return output



# class AttentionRecurrent(Recurrent):
#     def __init__(self, output_dim, init='glorot_uniform', truncate_gradient=-1,  inner_init='orthogonal',**kwargs):
#         self.output_dim = output_dim
#
#         self.init = initializations.get(init)
#         self.inner_init = initializations.get(inner_init)
#         self.inner_activation = activations.get("softmax")
#         self.truncate_gradient = truncate_gradient
#
#         super(AttentionRecurrent, self).__init__(**kwargs)
#
#
#     def build(self, input_shape):
#         input_dim = input_shape[2]
#
#         self.W = self.inner_init((input_dim, 2))
#         self.b = shared_zeros((2))
#         self.W_back = self.inner_init((input_dim, 2))
#         self.b_back = shared_zeros((2))
#
#         self.trainable_weights = [
#             self.W, self.b, self.W_back, self.b_back
#         ]
#
#
#
#
#
#     def _step(self,
#               x_mem, x_att,
#               h_tm1
#               ):
#         z0 = x_att[:, 0].dimshuffle(0, "x")
#         z1 = x_att[:, 1].dimshuffle(0, "x")
#
#         h_t = z0 * h_tm1 + z1 * x_mem
#         return h_t
#
#     def call(self, x, mask=None):
#         X = x
#
#         x_mem = X.dimshuffle((1, 0, 2))
#
#         x_att = self.inner_activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)
#
#         outputs, updates = theano.scan(
#             self._step,
#             sequences=[x_mem, x_att],
#             outputs_info=[
#                 T.unbroadcast(alloc_zeros_matrix(X.shape[0], self.output_dim), 1)
#             ],
#             truncate_gradient=self.truncate_gradient,
#             go_backwards=False)
#
#
#         outputs_bac, updates = theano.scan(
#             self._step,
#             sequences=[x_mem, self.inner_activation(T.dot(X.dimshuffle(1, 0, 2), self.W_back) + self.b_back)
# ],
#             outputs_info=[
#                 T.unbroadcast(alloc_zeros_matrix(X.shape[0], self.output_dim), 1)
#             ],
#             truncate_gradient=self.truncate_gradient,
#             go_backwards=True)
#
#         return outputs[-1] + outputs_bac[-1]
#
