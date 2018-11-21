using System;
using Xunit;
using KohonenNetwork;
using System.Linq;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;

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