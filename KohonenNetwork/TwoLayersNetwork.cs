using System.Collections.Generic;
using NeuralNetworkConstructor.Networks;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork
{
    public abstract class TwoLayersNetwork : Network
    {

        protected TwoLayersNetwork(IInputLayer inputLayer, ILayer<INotInputNode> outputLayer)
            : base(inputLayer, outputLayer)
        {
            _outputLayer = outputLayer;
        }

        protected readonly ILayer<INotInputNode> _outputLayer = new Layer();

        public override ICollection<ILayer<INotInputNode>> Layers => new ILayer<INotInputNode>[] { _outputLayer };

        public override ILayer<INotInputNode> OutputLayer => _outputLayer;

    }
}