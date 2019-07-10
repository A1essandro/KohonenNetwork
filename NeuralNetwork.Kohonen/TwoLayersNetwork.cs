using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Networks;
using NeuralNetwork.Structure.Nodes;
using System.Collections.Generic;

namespace NeuralNetwork.Kohonen
{
    public abstract class TwoLayersNetwork<TLayer> : Network
        where TLayer : IReadOnlyLayer<INotInputNode>
    {

        protected TwoLayersNetwork(IReadOnlyLayer<IMasterNode> inputLayer, TLayer outputLayer)
            : base(inputLayer, outputLayer)
        {
            _outputLayer = outputLayer;
        }

        protected readonly IReadOnlyLayer<INotInputNode> _outputLayer;

        public override ICollection<IReadOnlyLayer<INotInputNode>> Layers => new IReadOnlyLayer<INotInputNode>[] { _outputLayer };

        public override IReadOnlyLayer<INotInputNode> OutputLayer => _outputLayer;

        public TLayer OutputProjection => (TLayer)_outputLayer;

    }
}