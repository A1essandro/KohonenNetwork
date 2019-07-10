using KohonenNetwork;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using System;
using System.Linq;
using Xunit;

namespace Test
{
    public class NetworkTest
    {

        [Fact]
        public void LayersTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 5);
            var outputLayer = new Layer(() => new Neuron(), 2);
            var network = new KohonenNetwork.KohonenNetwork(inputLayer, outputLayer);

            Assert.Equal(1, network.Layers.Count());
            Assert.Throws<NotSupportedException>(() => network.Layers.Add(new Layer()));
            Assert.Equal(1, network.Layers.Count());
            Assert.True(network.Layers != network.Layers);
        }

        [Fact]
        public void ProjectionTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 5);
            var outputLayer = new Layer2D<INotInputNode>(() => new Neuron(), 6, 3, 2);
            var network = new KohonenNetwork<Layer2D<INotInputNode>>(inputLayer, outputLayer);

            Assert.Equal(3, network.OutputProjection.Projection.GetLength(0));
            Assert.Equal(2, network.OutputProjection.Projection.GetLength(1));
        }

        [Fact]
        public void NodesTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 5);
            var outputLayer = new Layer(() => new Neuron(), 2);
            var network = new KohonenNetwork.KohonenNetwork(inputLayer, outputLayer);

            Assert.Equal(5, network.InputLayer.Nodes.Count());
            Assert.Equal(2, network.OutputLayer.Nodes.Count());
        }
    }
}