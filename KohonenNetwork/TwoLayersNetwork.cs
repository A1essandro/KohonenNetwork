using System.Collections.Generic;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;

namespace KohonenNetwork
{
    public class TwoLayersNetwork : Network
    {
        protected readonly ILayer<IInputNode> _inputLayer = new InputLayer();
        protected readonly ILayer<INode> _outputLayer = new Layer();

        public override ICollection<ILayer<INode>> Layers => new ILayer<INode>[] { _outputLayer };

        public override ILayer<INode> OutputLayer => _outputLayer;

    }
}