using System.Collections.Generic;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;

namespace KohonenNetwork
{
    public abstract class TwoLayersNetwork : Network
    {

        protected TwoLayersNetwork(IInputLayer inputLayer, ILayer<INode> outputLayer)
            : base(inputLayer, outputLayer)
        {
            _outputLayer = outputLayer;
        }

        protected readonly ILayer<INode> _outputLayer = new Layer();

        public override ICollection<ILayer<INode>> Layers => new ILayer<INode>[] { _outputLayer };

        public override ILayer<INode> OutputLayer => _outputLayer;

    }
}