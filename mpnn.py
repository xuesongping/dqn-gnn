import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()

        self.hparams = hparams

        # Define layers for links
        self.Message = keras.Sequential([
            keras.layers.Dense(self.hparams['link_state_dim'], activation=tf.nn.selu, name="EdgeFirstLayer")
        ])

        self.Update = keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32, name="EdgeUpdate")

        # Define layers for nodes (with a size of `link_state_dim` - 5)
        self.NodeMessage = keras.Sequential([
            keras.layers.Dense(self.hparams['link_state_dim'] - 5, activation=tf.nn.selu, name="NodeFirstLayer")
        ])

        self.NodeUpdate = keras.layers.GRUCell(self.hparams['link_state_dim'] - 5, dtype=tf.float32, name="NodeUpdate")

        # Define readout layer
        self.Readout = keras.Sequential([
            # Note: Modify the input size based on how you combine node and link states
            keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu,
                               kernel_regularizer=regularizers.l2(hparams['l2']), name="Readout1"),
            keras.layers.Dropout(rate=hparams['dropout_rate']),
            keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu,
                               kernel_regularizer=regularizers.l2(hparams['l2']), name="Readout2"),
            keras.layers.Dropout(rate=hparams['dropout_rate']),
            keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']), name="Readout3")
        ])

    def build(self, input_shape=None):
        # Note: Provide appropriate input shapes for NodeMessage and NodeUpdate layers if necessary
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']]))
        self.NodeMessage.build(input_shape=tf.TensorShape([None, (self.hparams['link_state_dim'] - 5)*2]))
        self.NodeUpdate.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim'] - 5]))
        self.built = True

    @tf.function
    def call(self, graph_state, states_graph_ids, states_first, states_second, node_state, states_num_edges,
             training=False):
        # Define the forward pass for links
        link_state = graph_state
        for _ in range(self.hparams['T']):
            # Message passing for links
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)
            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            edge_outputs = self.Message(edgesConcat)
            edges_inputs = tf.math.unsorted_segment_sum(data=edge_outputs, segment_ids=states_second,
                                                        num_segments=states_num_edges)
            # GRUCell needs a 2D tensor as state, reshape the link_state before passing it
            link_state_shape = link_state.get_shape().as_list()
            link_state, _ = self.Update(edges_inputs,
                                        [tf.reshape(link_state, [link_state_shape[0], link_state_shape[1]])])

        # Define the forward pass for nodes (similar to links)
        for _ in range(self.hparams['T']):
            # Message passing for nodes
            mainNodes = tf.gather(node_state, states_first)
            neighNodes = tf.gather(node_state, states_second)
            nodesConcat = tf.concat([mainNodes, neighNodes], axis=1)

            node_outputs = self.NodeMessage(nodesConcat)
            nodes_inputs = tf.math.unsorted_segment_sum(data=node_outputs, segment_ids=states_second,
                                                        num_segments=states_num_edges)
            # Update the node state
            node_state_shape = node_state.get_shape().as_list()
            node_state, _ = self.NodeUpdate(nodes_inputs,
                                            [tf.reshape(node_state, [node_state_shape[0], node_state_shape[1]])])

        # Combine link and node states
        combined_state = tf.concat([link_state, node_state], axis=1)
        # Sum up states for each graph
        combined_state = tf.math.unsorted_segment_sum(combined_state, states_graph_ids,
                                                      tf.reduce_max(states_graph_ids) + 1)

        # Readout to produce graph-level output
        r = self.Readout(combined_state, training=training)
        return r