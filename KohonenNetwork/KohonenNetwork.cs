using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using NeuralNetworkConstructor.Network.Node.Synapse;

namespace KohonenNetwork
{
    public class KohonenNetwork<TFunc> : Network
        where TFunc : IActivationFunction, new()
    {

        private ILayer<IInputNode> _inputLayer = new InputLayer();
        private ILayer<INode> _outputLayer = new Layer();

        public override ICollection<ILayer<INode>> Layers => new ILayer<INode>[] { _outputLayer };

        public override ILayer<INode> OutputLayer => _outputLayer;

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = true)
        {
            for (var i = 0; i < inputNodes; i++)
            {
                _inputLayer.Nodes.Add(new InputNode());
            }

            if (withBias)
            {
                _inputLayer.Nodes.Add(new InputBias());
            }

            for (var i = 0; i < inputNodes; i++)
            {
                _outputLayer.Nodes.Add(new Neuron<TFunc>());
            }

            Synapse.Generator.EachToEach(_inputLayer, _outputLayer);
        }

    }
}
