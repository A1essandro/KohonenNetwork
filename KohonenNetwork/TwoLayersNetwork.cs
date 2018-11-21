using System.Collections.Generic;
using NeuralNetworkConstructor.Networks;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork
{
    public abstract class TwoLayersNetwork : Network
    {

        protected TwoLayersNetwork(IReadOnlyLayer<IMasterNode> inputLayer, IReadOnlyLayer<INotInputNode> outputLayer)
            : base(inputLayer, outputLayer)
        {
            _outputLayer = outputLayer;
        }

        protected readonly IReadOnlyLayer<INotInputNode> _outputLayer;

        public override ICollection<IReadOnlyLayer<INotInputNode>> Layers => new IReadOnlyLayer<INotInputNode>[] { _outputLayer };

        public override IReadOnlyLayer<INotInputNode> OutputLayer => _outputLayer;

    }
}